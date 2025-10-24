use std::mem::swap;
use std::vec::Vec;

/// The indexed bitmap.
///
/// The core of this data structure is a bitmap, but we build
/// index to accelerate search operations for zeroes and ones.
pub struct IndexedBitmap {
    bitmap: Vec<u128>,
    index: Vec<Vec<u128>>,
}

impl IndexedBitmap {
    pub fn new() -> IndexedBitmap {
        IndexedBitmap {
            bitmap: Vec::new(),
            index: Vec::new(),
        }
    }

    fn get_previous_layer(&self, layer: usize) -> &Vec<u128> {
        if layer == 0 {
            &self.bitmap
        } else {
            &self.index[layer - 1]
        }
    }

    fn reverse_n_bits(&mut self, n: usize) {
        let num_bitmap_pages = (n + 127) >> 7;
        let num_alloc_pages = self.bitmap.len();
        if num_alloc_pages >= num_bitmap_pages {
            return;
        }
        self.bitmap.resize(num_bitmap_pages, 0);
        // end_page is the highest page of previous layer that
        // has one in it. Since after resizing the pages are
        // filled with zero automatically, and zero pages
        // corresponds to 0b00 index in the upper layer, so
        // we just need to care about filling in the gap from
        // the allocated pages to the last page with set bit.
        let mut end_pages = self.page_size();
        let mut layer = 0usize;
        loop {
            let pages = self.get_previous_layer(layer).len();
            // We won't create a new layer of index unless the
            // previous layer has already contained more
            // than 1 page.
            if pages <= 1 {
                return;
            }
            if self.index.len() < layer + 1 {
                self.index.push(Vec::new());
            }
            // By swapping it out, we will no longer need to
            // care about fighting over the mutable borrow of self.
            // All we need to do is to swap it back when we are
            // done with the index construction.
            let mut current = Vec::<u128>::new();
            swap(&mut self.index[layer], &mut current);
            // Since reverse bits is nothing more than appending
            // zeroes, and the existing pages has already built as
            // if the num presenting pages are all zeroes, we can
            // safely ignore those pages.
            let start_pages = current.len() << 6;
            current.resize((pages + 63) >> 6, 0);
            let previous = self.get_previous_layer(layer);
            for i in start_pages..end_pages {
                let p = i >> 6;
                let off = i & 63;
                let mut bits = 0u128;
                let target = previous[i];
                bits |= if target != 0u128 { 1 } else { 0 };
                bits |= if target == u128::MAX { 2 } else { 0 };
                let mut index = current[p];
                let masks = 3u128 << (off << 1);
                index |= masks;
                index ^= masks;
                index |= bits << (off << 1);
                current[p] = index;
            }
            swap(&mut self.index[layer], &mut current);
            layer += 1;
            end_pages = (end_pages + 63) >> 6;
        }
    }

    fn refresh_index_page(&mut self, bitmap_page: usize) {
        let mut page = bitmap_page;
        let num_layers = self.index.len();
        for layer in 0..num_layers {
            let p = page >> 6;
            let off = page & 63;
            let old_index = self.index[layer][p];
            let mut bits = 0u128;
            let target = self.get_previous_layer(layer)[page];
            bits |= if target != 0u128 { 1 } else { 0 };
            bits |= if target == u128::MAX { 2 } else { 0 };
            let mut new_index = old_index;
            let masks = 3u128 << (off << 1);
            new_index |= masks;
            new_index ^= masks;
            new_index |= bits << (off << 1);
            if new_index == old_index {
                break;
            }
            self.index[layer][p] = new_index;
            page = p;
        }
    }

    pub fn bitset(&mut self, n: usize, set: bool) {
        self.reverse_n_bits(n + 1);
        let p = n >> 7;
        let off = n & 127;
        let page = self.bitmap[p];
        let mask = 1u128 << off;
        let new_page = (page | mask) ^ (if set { 0 } else { mask });
        if page != new_page {
            self.bitmap[p] = new_page;
            self.refresh_index_page(p);
        }
    }

    pub fn bitget(&self, n: usize) -> bool {
        let p = n >> 7;
        let off = n & 127;
        if p >= self.bitmap.len() {
            false
        } else {
            (self.bitmap[p] & (1 << off)) != 0
        }
    }

    fn lowest_zero_page(&self) -> usize {
        let mut page = 0;
        let num_layers = self.index.len();
        for i in 0..num_layers {
            let layer = num_layers - i - 1;
            if page >= self.index[layer].len() {
                page <<= 6;
                continue;
            }
            let mut next_page = page << 6;
            let mut target = self.index[layer][page];
            if target == u128::MAX {
                next_page += 128;
            } else {
                target = target ^ target.wrapping_add(1);
                target >>= 1;
                target = target.wrapping_add(1);
                next_page += (target.trailing_zeros() >> 1) as usize;
            }
            page = next_page;
        }
        page
    }

    pub fn lowest_zero(&self) -> usize {
        let page = self.lowest_zero_page();
        if page >= self.bitmap.len() {
            self.bitmap.len() << 7
        } else {
            let mut result = page << 7;
            let mut target = self.bitmap[page];
            if target == u128::MAX {
                result + 128
            } else {
                target = target ^ target.wrapping_add(1);
                target >>= 1;
                target = target.wrapping_add(1);
                result += target.trailing_zeros() as usize;
                result
            }
        }
    }

    fn highest_one_page(&self) -> Option<usize> {
        let mut page = 0;
        let num_layers = self.index.len();
        for i in 0..num_layers {
            let layer = num_layers - i - 1;
            if page >= self.index[layer].len() {
                return None;
            }
            let mut next_page = page << 6;
            let mut target = self.index[layer][page];
            // We just need the non-zero bit of every pages.
            // target &= 0b0101_..._01; // 0x5 = 0b0101
            // 16bit x  8    7    6    5    4    3    2    1
            target &= 0x5555_5555_5555_5555_5555_5555_5555_5555;
            target |= target << 1;
            let off = 64 - (target.leading_zeros() >> 1) as usize;
            if off == 0 {
                // This means this page is the whole zero
                // 0.leading_zero() == 128, 64 - (128 >> 1) = 64 - 64 = 0
                // Whatever way we get here, there's no
                // way we can go further, so we return None.
                return None;
            }
            next_page += off - 1;
            page = next_page;
        }
        if page < self.bitmap.len() && self.bitmap[page] != 0 {
            Some(page)
        } else {
            None
        }
    }

    fn page_size(&self) -> usize {
        self.highest_one_page().map_or(0, |x| x + 1)
    }

    pub fn highest_one(&self) -> Option<usize> {
        if let Some(page) = self.highest_one_page() {
            let target = self.bitmap[page];
            // This is guaranteed by highest_one_page, as it must
            // return None if all pages are zero.
            assert!(target != 0);
            Some((page << 7) + (127 - target.leading_zeros()) as usize)
        } else {
            None
        }
    }

    /// Size measured in bits to accomodate the bitmap.
    ///
    /// It is set to self.highest_one() + 1 if there's any
    /// set bit, or 0 if there's no set bit.
    pub fn size(&self) -> usize {
        self.highest_one().map_or(0, |x| x + 1)
    }

    fn lowest_one_page(&self) -> Option<usize> {
        let mut page = 0;
        let num_layers = self.index.len();
        for i in 0..num_layers {
            let layer = num_layers - i - 1;
            if page >= self.index[layer].len() {
                return None;
            }
            let mut next_page = page << 6;
            let target = self.index[layer][page];
            let off = (target.trailing_zeros() >> 1) as usize;
            next_page += off;
            page = next_page;
        }
        if page < self.bitmap.len() && self.bitmap[page] != 0 {
            Some(page)
        } else {
            None
        }
    }

    pub fn lowest_one(&self) -> Option<usize> {
        if let Some(page) = self.lowest_one_page() {
            let target = self.bitmap[page];
            assert!(target != 0);
            Some((page << 7) + target.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operations_1() {
        let mut bitmap = IndexedBitmap::new();
        assert_eq!(bitmap.lowest_zero(), 0);
        assert_eq!(bitmap.lowest_one(), None);
        assert_eq!(bitmap.highest_one(), None);
        assert_eq!(bitmap.size(), 0);

        // The bit set is <128, which does not trigger
        // allocation of the index pages.
        bitmap.bitset(0, true);
        bitmap.bitset(1, true);
        assert_eq!(bitmap.bitmap.len(), 1);
        assert_eq!(bitmap.bitmap[0], 0b011);
        assert_eq!(bitmap.index.len(), 0);
        assert_eq!(bitmap.lowest_zero(), 2);
        assert_eq!(bitmap.lowest_one(), Some(0));
        assert_eq!(bitmap.highest_one(), Some(1));
        assert_eq!(bitmap.size(), 2);

        // The bit set is >=128, which triggers an
        // allocation of the index pages.
        bitmap.bitset(128, true);
        assert_eq!(bitmap.bitmap.len(), 2);
        assert_eq!(bitmap.bitmap[0], 0b011);
        assert_eq!(bitmap.bitmap[1], 0b01);
        assert_eq!(bitmap.index.len(), 1);
        assert_eq!(bitmap.index[0].len(), 1);
        assert_eq!(bitmap.index[0][0], 0b0101);
        assert_eq!(bitmap.lowest_zero(), 2);
        assert_eq!(bitmap.lowest_one(), Some(0));
        assert_eq!(bitmap.highest_one(), Some(128));
        assert_eq!(bitmap.size(), 129);

        // Now clear the 128 bit. Without explicit invocation
        // to the shrink_to_fit function, it will just clear
        // the higher bit and leave the index page there.
        bitmap.bitset(128, false);
        assert_eq!(bitmap.bitmap.len(), 2);
        assert_eq!(bitmap.bitmap[0], 0b011);
        assert_eq!(bitmap.bitmap[1], 0b00);
        assert_eq!(bitmap.index.len(), 1);
        assert_eq!(bitmap.index[0].len(), 1);
        assert_eq!(bitmap.index[0][0], 0b01);
        assert_eq!(bitmap.lowest_zero(), 2);
        assert_eq!(bitmap.lowest_one(), Some(0));
        assert_eq!(bitmap.highest_one(), Some(1));
        assert_eq!(bitmap.size(), 2);

        // Clear the first bit so that the lowest
        // zero is the first bit.
        bitmap.bitset(1, false);
        assert_eq!(bitmap.lowest_zero(), 1);
        assert_eq!(bitmap.lowest_one(), Some(0));
        assert_eq!(bitmap.highest_one(), Some(0));
        assert_eq!(bitmap.size(), 1);

        // Clear all bits but without shrinking,
        // to see whether it works normally.
        bitmap.bitset(0, false);
        assert_eq!(bitmap.lowest_zero(), 0);
        assert_eq!(bitmap.lowest_one(), None);
        assert_eq!(bitmap.highest_one(), None);
        assert_eq!(bitmap.size(), 0);
    }

    #[test]
    fn test_operations_2() {
        let mut bitmap = IndexedBitmap::new();
        let upper_bound = 300000usize;
        assert_eq!(bitmap.lowest_zero(), 0);
        assert_eq!(bitmap.lowest_one(), None);
        assert_eq!(bitmap.highest_one(), None);
        for i in 0..upper_bound {
            bitmap.bitset(i, true);
            assert_eq!(bitmap.lowest_zero(), i + 1);
            assert_eq!(bitmap.lowest_one(), Some(0));
            assert_eq!(bitmap.highest_one(), Some(i));
        }
        assert_eq!(bitmap.lowest_zero(), upper_bound);
        assert_eq!(bitmap.lowest_one(), Some(0));
        assert_eq!(bitmap.highest_one(), Some(upper_bound - 1));
        for i in 0..upper_bound {
            bitmap.bitset(i, false);
            assert_eq!(bitmap.lowest_zero(), i);
            assert_eq!(bitmap.lowest_one(), Some(if i == 0 { 1 } else { 0 }),);
            assert_eq!(
                bitmap.highest_one(),
                Some(upper_bound - 2 + (if i == upper_bound - 1 { 0 } else { 1 })),
            );
            bitmap.bitset(i, true);
            assert_eq!(bitmap.lowest_zero(), upper_bound);
            assert_eq!(bitmap.lowest_one(), Some(0));
            assert_eq!(bitmap.highest_one(), Some(upper_bound - 1),);
        }
        for i in 0..upper_bound {
            let target = upper_bound - i - 1;
            bitmap.bitset(target, false);
            assert_eq!(bitmap.lowest_zero(), target);
            assert_eq!(
                bitmap.lowest_one(),
                if target == 0 { None } else { Some(0) },
            );
            assert_eq!(bitmap.size(), target);
        }
    }

    #[test]
    fn test_operations_3() {
        let mut bitmap = IndexedBitmap::new();
        assert_eq!(bitmap.lowest_one(), None);
        assert_eq!(bitmap.highest_one(), None);
        bitmap.bitset(0, true);
        bitmap.bitset(0, false);
        assert_eq!(bitmap.lowest_one(), None);
        assert_eq!(bitmap.highest_one(), None);
        for i in 0..32 {
            bitmap.bitset(1 << i, true);
            assert_eq!(bitmap.lowest_one(), Some(1 << i));
            assert_eq!(bitmap.highest_one(), Some(1 << i));
            bitmap.bitset(1 << i, false);
            assert_eq!(bitmap.lowest_one(), None);
            assert_eq!(bitmap.highest_one(), None);
        }
    }
}
