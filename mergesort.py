from bisect import bisect


def mergesort(sequence, start, end):
    if start == end:
        return [sequence[start]]
    else:
        mid = int((start+end)/2)
        sq_left = mergesort(sequence, start, mid)
        sq_right = mergesort(sequence, mid+1, end)
        merger = []
        low_pos, insert_pos = 0, 0
        l_left, l_right = mid-start+1, end-mid
        FLAG = False
        for i, ele in enumerate(sq_right):
            if low_pos >= l_left:
                FLAG = True
                break
            insert_pos = bisect(sq_left, ele, lo=low_pos)
            merger += (sq_left[low_pos:insert_pos] + [ele])
            low_pos = insert_pos

        if low_pos < l_left:
            merger += sq_left[low_pos:]
        elif FLAG:
            merger += sq_right[i:]

        return merger


if __name__ == '__main__':
    sq = list(range(10, 20)) + list(range(5, 10)) + list(range(5))
    merger = mergesort(sq, 0, len(sq)-1)
    print(merger)
