//! `std::sort` of the C++ standard library the reference runtime is built with, reproduced element for element.
//!
//! llama.cpp selects the experts of a mixture-of-experts layer with `std::sort` (an unstable sort) over the router
//! scores. When two scores are exactly equal, the order of the tied experts, and so which of them is selected, is
//! decided by the sort algorithm of the standard library. To reproduce the reference's choice the same algorithm is
//! used here: the one of Microsoft's STL on Windows (the reference is built with clang against the MSVC STL) and the
//! one of libstdc++ elsewhere (the Linux reference). Both are introsorts and order the same values identically as
//! long as there are no ties.

/// Sorts `v` with the comparator `less` (strict weak order) exactly like the reference's `std::sort`.
pub fn reference_sort<T: Copy>(v: &mut [T], less: &impl Fn(T, T) -> bool) {
    if cfg!(target_env = "msvc") {
        msvc::sort(v, less);
    } else {
        libstdcxx::sort(v, less);
    }
}

/// Microsoft STL (`<algorithm>`, `_Sort_unchecked`): insertion sort up to 32 elements, Tukey's ninther above 40, a
/// three-way partition around the pivot, heap sort after `1.5 log2(n)` divisions.
pub mod msvc {
    const ISORT_MAX: usize = 32;

    pub fn sort<T: Copy>(v: &mut [T], less: &impl Fn(T, T) -> bool) {
        let n = v.len();
        sort_unchecked(v, 0, n, n as isize, less);
    }

    fn sort_unchecked<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, mut ideal: isize, less: &impl Fn(T, T) -> bool) {
        loop {
            if last - first <= ISORT_MAX {
                insertion_sort(v, first, last, less);
                return;
            }
            if ideal <= 0 {
                make_heap(v, first, last, less);
                sort_heap(v, first, last, less);
                return;
            }
            let (mid_first, mid_last) = partition_by_median_guess(v, first, last, less);
            ideal = (ideal >> 1) + (ideal >> 2);
            if mid_first - first < last - mid_last {
                sort_unchecked(v, first, mid_first, ideal, less);
                first = mid_last;
            } else {
                sort_unchecked(v, mid_last, last, ideal, less);
                last = mid_first;
            }
        }
    }

    fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        if first == last {
            return;
        }
        let mut mid = first + 1;
        while mid != last {
            let val = v[mid];
            if less(val, v[first]) {
                v.copy_within(first..mid, first + 1);
                v[first] = val;
            } else {
                let mut hole = mid;
                let mut prev = hole - 1;
                while less(val, v[prev]) {
                    v[hole] = v[prev];
                    hole = prev;
                    prev -= 1;
                }
                v[hole] = val;
            }
            mid += 1;
        }
    }

    fn med3<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        if less(v[mid], v[first]) {
            v.swap(mid, first);
        }
        if less(v[last], v[mid]) {
            v.swap(last, mid);
            if less(v[mid], v[first]) {
                v.swap(mid, first);
            }
        }
    }

    fn guess_median<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        let count = last - first;
        if 40 < count {
            let step = (count + 1) >> 3;
            let two_step = step << 1;
            med3(v, first, first + step, first + two_step, less);
            med3(v, mid - step, mid, mid + step, less);
            med3(v, last - two_step, last - step, last, less);
            med3(v, first + step, mid, last - step, less);
        } else {
            med3(v, first, mid, last, less);
        }
    }

    fn partition_by_median_guess<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) -> (usize, usize) {
        let mid = first + ((last - first) >> 1);
        guess_median(v, first, mid, last - 1, less);
        partition_by_pivot(v, first, mid, last, less)
    }

    fn partition_by_pivot<T: Copy>(v: &mut [T], first: usize, mut pfirst: usize, last: usize, less: &impl Fn(T, T) -> bool) -> (usize, usize) {
        let mut plast = pfirst + 1;
        while first < pfirst && !less(v[pfirst - 1], v[pfirst]) && !less(v[pfirst], v[pfirst - 1]) {
            pfirst -= 1;
        }
        while plast < last && !less(v[plast], v[pfirst]) && !less(v[pfirst], v[plast]) {
            plast += 1;
        }
        let mut gfirst = plast;
        let mut glast = pfirst;
        loop {
            while gfirst < last {
                if less(v[pfirst], v[gfirst]) {
                } else if less(v[gfirst], v[pfirst]) {
                    break;
                } else if plast != gfirst {
                    v.swap(plast, gfirst);
                    plast += 1;
                } else {
                    plast += 1;
                }
                gfirst += 1;
            }
            while first < glast {
                let gp = glast - 1;
                if less(v[gp], v[pfirst]) {
                } else if less(v[pfirst], v[gp]) {
                    break;
                } else {
                    pfirst -= 1;
                    if pfirst != gp {
                        v.swap(pfirst, gp);
                    }
                }
                glast -= 1;
            }
            if glast == first && gfirst == last {
                return (pfirst, plast);
            }
            if glast == first {
                if plast != gfirst {
                    v.swap(pfirst, plast);
                }
                plast += 1;
                v.swap(pfirst, gfirst);
                pfirst += 1;
                gfirst += 1;
            } else if gfirst == last {
                glast -= 1;
                pfirst -= 1;
                if glast != pfirst {
                    v.swap(glast, pfirst);
                }
                plast -= 1;
                v.swap(pfirst, plast);
            } else {
                glast -= 1;
                v.swap(gfirst, glast);
                gfirst += 1;
            }
        }
    }

    fn push_heap_by_index<T: Copy>(v: &mut [T], base: usize, mut hole: usize, top: usize, val: T, less: &impl Fn(T, T) -> bool) {
        while top < hole {
            let idx = (hole - 1) >> 1;
            if !less(v[base + idx], val) {
                break;
            }
            v[base + hole] = v[base + idx];
            hole = idx;
        }
        v[base + hole] = val;
    }

    fn pop_heap_hole_by_index<T: Copy>(v: &mut [T], base: usize, mut hole: usize, bottom: usize, val: T, less: &impl Fn(T, T) -> bool) {
        let top = hole;
        let mut idx = hole;
        let max_non_leaf = (bottom - 1) >> 1;
        while idx < max_non_leaf {
            idx = 2 * idx + 2;
            if less(v[base + idx], v[base + idx - 1]) {
                idx -= 1;
            }
            v[base + hole] = v[base + idx];
            hole = idx;
        }
        if idx == max_non_leaf && bottom % 2 == 0 {
            v[base + hole] = v[base + bottom - 1];
            hole = bottom - 1;
        }
        push_heap_by_index(v, base, hole, top, val, less);
    }

    fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        let bottom = last - first;
        let mut hole = bottom >> 1;
        while hole > 0 {
            hole -= 1;
            let val = v[first + hole];
            pop_heap_hole_by_index(v, first, hole, bottom, val, less);
        }
    }

    fn sort_heap<T: Copy>(v: &mut [T], first: usize, mut last: usize, less: &impl Fn(T, T) -> bool) {
        while last - first >= 2 {
            last -= 1;
            let val = v[last];
            v[last] = v[first];
            pop_heap_hole_by_index(v, first, 0, last - first, val, less);
        }
    }
}

/// libstdc++ (`bits/stl_algo.h`, `std::__sort`): introsort down to 16 elements with a median-of-three pivot moved to
/// the front and an unguarded partition, heap sort at depth `2 log2(n)`, then a final insertion sort.
pub mod libstdcxx {
    const THRESHOLD: usize = 16;

    pub fn sort<T: Copy>(v: &mut [T], less: &impl Fn(T, T) -> bool) {
        let n = v.len();
        if n == 0 {
            return;
        }
        let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as usize;
        introsort_loop(v, 0, n, depth, less);
        final_insertion_sort(v, 0, n, less);
    }

    fn introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, less: &impl Fn(T, T) -> bool) {
        while last - first > THRESHOLD {
            if depth == 0 {
                make_heap(v, first, last, less);
                sort_heap(v, first, last, less);
                return;
            }
            depth -= 1;
            let cut = unguarded_partition_pivot(v, first, last, less);
            introsort_loop(v, cut, last, depth, less);
            last = cut;
        }
    }

    fn move_median_to_first<T: Copy>(v: &mut [T], result: usize, a: usize, b: usize, c: usize, less: &impl Fn(T, T) -> bool) {
        if less(v[a], v[b]) {
            if less(v[b], v[c]) {
                v.swap(result, b);
            } else if less(v[a], v[c]) {
                v.swap(result, c);
            } else {
                v.swap(result, a);
            }
        } else if less(v[a], v[c]) {
            v.swap(result, a);
        } else if less(v[b], v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, b);
        }
    }

    fn unguarded_partition_pivot<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) -> usize {
        let mid = first + (last - first) / 2;
        move_median_to_first(v, first, first + 1, mid, last - 1, less);
        let pivot = first;
        let (mut lo, mut hi) = (first + 1, last);
        loop {
            while less(v[lo], v[pivot]) {
                lo += 1;
            }
            hi -= 1;
            while less(v[pivot], v[hi]) {
                hi -= 1;
            }
            if lo >= hi {
                return lo;
            }
            v.swap(lo, hi);
            lo += 1;
        }
    }

    fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, less: &impl Fn(T, T) -> bool) {
        let val = v[last];
        let mut next = last - 1;
        while less(val, v[next]) {
            v[last] = v[next];
            last = next;
            next -= 1;
        }
        v[last] = val;
    }

    fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        if first == last {
            return;
        }
        for i in first + 1..last {
            if less(v[i], v[first]) {
                let val = v[i];
                v.copy_within(first..i, first + 1);
                v[first] = val;
            } else {
                unguarded_linear_insert(v, i, less);
            }
        }
    }

    fn final_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        if last - first > THRESHOLD {
            insertion_sort(v, first, first + THRESHOLD, less);
            for i in first + THRESHOLD..last {
                unguarded_linear_insert(v, i, less);
            }
        } else {
            insertion_sort(v, first, last, less);
        }
    }

    fn push_heap<T: Copy>(v: &mut [T], base: usize, mut hole: usize, top: usize, val: T, less: &impl Fn(T, T) -> bool) {
        let mut parent = hole.wrapping_sub(1) / 2;
        while hole > top && less(v[base + parent], val) {
            v[base + hole] = v[base + parent];
            hole = parent;
            parent = hole.wrapping_sub(1) / 2;
        }
        v[base + hole] = val;
    }

    fn adjust_heap<T: Copy>(v: &mut [T], base: usize, mut hole: usize, len: usize, val: T, less: &impl Fn(T, T) -> bool) {
        let top = hole;
        let mut second = hole;
        while second < len.saturating_sub(1) / 2 {
            second = 2 * (second + 1);
            if less(v[base + second], v[base + second - 1]) {
                second -= 1;
            }
            v[base + hole] = v[base + second];
            hole = second;
        }
        if len & 1 == 0 && len >= 2 && second == (len - 2) / 2 {
            second = 2 * (second + 1);
            v[base + hole] = v[base + second - 1];
            hole = second - 1;
        }
        push_heap(v, base, hole, top, val, less);
    }

    fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(T, T) -> bool) {
        let len = last - first;
        if len < 2 {
            return;
        }
        let mut parent = (len - 2) / 2;
        loop {
            let val = v[first + parent];
            adjust_heap(v, first, parent, len, val, less);
            if parent == 0 {
                return;
            }
            parent -= 1;
        }
    }

    fn sort_heap<T: Copy>(v: &mut [T], first: usize, mut last: usize, less: &impl Fn(T, T) -> bool) {
        while last - first > 1 {
            last -= 1;
            let val = v[last];
            v[last] = v[first];
            adjust_heap(v, first, 0, last - first, val, less);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn both_sorts_order_like_a_reference_sort() {
        let mut seed = 3;
        for n in [0usize, 1, 2, 5, 16, 17, 31, 32, 33, 40, 41, 64, 128, 300, 1000] {
            for distinct in [3u64, 50, 1 << 30] {
                let vals: Vec<u64> = (0..n).map(|_| lcg(&mut seed) % distinct).collect();
                let less = |a: usize, b: usize| vals[a] > vals[b];
                let mut want: Vec<u64> = vals.clone();
                want.sort_by(|a, b| b.cmp(a));
                for f in [msvc::sort::<usize>, libstdcxx::sort::<usize>] {
                    let mut idx: Vec<usize> = (0..n).collect();
                    f(&mut idx, &less);
                    let got: Vec<u64> = idx.iter().map(|&i| vals[i]).collect();
                    assert_eq!(got, want, "n={n} distinct={distinct}");
                    let mut seen = idx.clone();
                    seen.sort();
                    assert_eq!(seen, (0..n).collect::<Vec<_>>());
                }
            }
        }
    }
}
