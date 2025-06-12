#!/usr/bin/env python3
import numpy as np
import torch


# redefine numpy version by changing the default atol from 0 to 1e-10
# this way we don't need to write atol=1e-10 in every call
# the reason for atol=1e-10 is because we want the assert to consider 1e-16 to be equal to 0 and not raise an exception in that case
def assert_allclose(actual, desired, rtol=1e-7, atol=1e-10, equal_nan=True,
                    err_msg='', verbose=True):
    return np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)


def cylinder_sdf_gold(p, r, h0, h1):
    '''
    p: [3]
    r, h0, h1: float
    Returns SDF of a cylinder at (0, 0, 0) with radius r and heights h0 and h1
    '''
    assert h1 >= h0

    px, py, pz = p
    hordist = np.hypot(px, pz) - r  # inside is negative
    verdist = max(py - h1, h0 - py)  # inside is negative
    if verdist < 0:
        res = max(hordist, verdist)
    elif hordist < 0:
        res = verdist
    else:
        # hordist > 0
        res = np.hypot(hordist, verdist)

    # if something is not handled, res is not assigned and this throws an exception
    return res


def cylinder_sdf(p, r, h0, h1):
    '''
    p: [..., 3]
    r, h0, h1: float
    Returns SDF of a cylinder at (0, 0, 0) with radius r and heights h0 and h1.
    Note that this version accepts multiple points p in the same call.
    '''
    assert h1 >= h0

    if isinstance(p, (tuple, list, np.ndarray)):
        print('warning: casting to torch.tensor')
        p = torch.tensor(p, dtype=torch.float64)

    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    hordist = torch.hypot(px, pz) - r  # inside is negative
    verdist = torch.maximum(py - h1, h0 - py)  # inside is negative

    mask_ver = verdist < 0

    res = mask_ver * torch.maximum(hordist, verdist)
    res = res + (~mask_ver) * torch.hypot(torch.relu(hordist), verdist)

    return res


def test_cylinder_sdf():
    assert_allclose(cylinder_sdf((0,   0, 0), 1, -1, 1), -1)
    assert_allclose(cylinder_sdf((0.5, 0, 0), 1, -1, 1), -0.5)
    assert_allclose(cylinder_sdf((1,   0, 0), 1, -1, 1), 0)
    assert_allclose(cylinder_sdf((1.5, 0, 0), 1, -1, 1), 0.5)
    assert_allclose(cylinder_sdf((2,   0, 0), 1, -1, 1), 1)

    assert_allclose(cylinder_sdf((0, 0, 0), 1,     0.1, 1), 0.1)
    assert_allclose(cylinder_sdf((0, 0, 0), 1,    -0.1, 1), -0.1)
    assert_allclose(cylinder_sdf((0, 0, 0), 0.05, -0.1, 1), -0.05)

    assert_allclose(cylinder_sdf((1, 0, 0),  1, -1, 1), 0)
    assert_allclose(cylinder_sdf((0, 1, 0),  1, -1, 1), 0)
    assert_allclose(cylinder_sdf((0, -1, 0), 1, -1, 1), 0)


def test_cylinder_sdf_vs_gold():
    for _ in range(100):
        p = np.random.random((3,))*5
        a = cylinder_sdf(p, 1, -2, 3)
        b = cylinder_sdf_gold(p, 1, -2, 3)
        assert_allclose(a, b)


def cylinder_intersect_gold(o, d, r, h0, h1):
    '''
    o, d: [3]
    r, h0, h1: float
    Intersects ray p(t)=o+td with the cylinder at (0, 0, 0) with radius r and heights h0 and h1, and returns the t's of the first and last intersections.
       If the ray is coming from inside the cylinder, then the first intersection is going to be the ray origin (t=0).
       If there is no intersection, then [+inf, -inf] is returned.
    '''

    EPS = 1e-6  # for torch.float32

    ox, oy, oz = o
    dx, dy, dz = d

    res = []

    # plane intersections
    if not np.allclose(dy, 0):
        t3 = (h0 - oy) / dy
        t4 = (h1 - oy) / dy

        res += [t3, t4]

    # side intersections
    A = dx**2+dz**2
    B = ox*dx+oz*dz
    C = ox**2+oz**2-r**2

    D = B**2-A*C
    if np.allclose(A, 0):
        if not np.allclose(B, 0):
            # should not happen
            # A=0 means dx=dz=0 and therefore B=0
            assert False
            # t1 = -C/B/2
            # print(A, B, C, t1)
            # assert_allclose(A*t1**2+2*B*t1+C, 0)
            # print('t1:', t1, np.array(o)+np.array(d)*t1)
            # res += [t1]
    elif D > 0:
        t1 = (-B-np.sqrt(D))/A
        t2 = (-B+np.sqrt(D))/A
        # print('A =', A, 'B =', B, 'C =', C, 't1 =', t1, 't2 =', t2)
        assert_allclose(A*t1**2+2*B*t1+C, 0)
        assert_allclose(A*t2**2+2*B*t2+C, 0)

        res += [t1, t2]


    # only keep the positive results, 0 not included
    res = [t for t in res if t > EPS]
    res.sort()

    # the origin is inside the cylinder
    if cylinder_sdf(o, r, h0, h1) < EPS:
        res = [0] + res

    res = [t for t in res if cylinder_sdf(o+d*t, r, h0, h1) < EPS]

    if len(res) > 0:
        return [np.min(res), np.max(res)]
    else:
        return [np.inf, -np.inf]


def cylinder_intersect(o, d, r, h0, h1):
    '''
    o, d: [..., 3]
    r, h0, h1: float
    Intersects rays p(t)=o+td with the cylinder at (0, 0, 0) with radius r and heights h0 and h1, and returns the t's of the first and last intersections.
    If the ray is coming from inside the cylinder, then the first intersection is going to be the ray origin (t=0).
    If there is no intersection, then [+inf, -inf] is returned.
    Note that this version accepts multiple rays o, d in the same call.
    '''

    if isinstance(o, (tuple, list, np.ndarray)):
        print('warning: casting to torch.tensor')
        o = torch.tensor(o, dtype=torch.float64)
        d = torch.tensor(d, dtype=torch.float64)

    # if o.dtype == torch.float16 or d.dtype == torch.float16:
    #     EPS = 5e-3  # for torch.float16
    # else:
    #     EPS = 1e-6  # for torch.float32
    EPS = 1e-6  # for torch.float32

    ox, oy, oz = o[..., 0], o[..., 1], o[..., 2]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]

    res = []

    # plane intersections
    # note: -inf is chosen for the reason that we ignore negative ts when collating the results
    # e.g., in this case, we don't want these results when dy=0, because they are not valid. neither inf, -inf, nan are the results we want and therefore we mask it out by replacing them with -inf
    t3 = torch.nan_to_num((h0 - oy) / dy, -torch.inf, -torch.inf, -torch.inf)
    t4 = torch.nan_to_num((h1 - oy) / dy, -torch.inf, -torch.inf, -torch.inf)

    p3 = o+d*t3[..., None]
    p4 = o+d*t4[..., None]

    t3mask = p3[..., 0]**2 + p3[..., 2]**2 < r + EPS  # todo: check if eps is needed
    t4mask = p4[..., 0]**2 + p4[..., 2]**2 < r + EPS  # todo: check if eps is needed

    t3 = t3.masked_fill(~t3mask, -torch.inf)
    t4 = t4.masked_fill(~t4mask, -torch.inf)

    res += [t3, t4]

    # side intersections
    A = dx**2+dz**2
    B = ox*dx+oz*dz
    C = ox**2+oz**2-r**2

    D = B**2-A*C

    # here, the same -inf trick works because for sqrt of negative numbers we get nan, and for division by zero we get inf. neither of the cases is valid and therefore is replaced with -inf. and every other remaining case is valid
    t1 = torch.nan_to_num((-B-torch.sqrt(D))/A, -torch.inf, -torch.inf, -torch.inf)
    t2 = torch.nan_to_num((-B+torch.sqrt(D))/A, -torch.inf, -torch.inf, -torch.inf)

    p1 = o+d*t1[..., None]
    p2 = o+d*t2[..., None]

    t1mask = (h0 - EPS < p1[..., 1]) & (p1[..., 1] < h1 + EPS)  # todo: check if eps is needed
    t2mask = (h0 - EPS < p2[..., 1]) & (p2[..., 1] < h1 + EPS)  # todo: check if eps is needed

    t1 = t1.masked_fill(~t1mask, -torch.inf)
    t2 = t2.masked_fill(~t2mask, -torch.inf)

    res += [t1, t2]

    # going to be 0 when origin is inside the cylinder, -inf otherwise
    # t5 = o.new_zeros(o.shape[:-1]).masked_fill(~(cylinder_sdf(o, r, h0, h1) < EPS), -torch.inf)
    t5 = o.new_full(o.shape[:-1], -torch.inf).masked_fill((cylinder_sdf(o, r, h0, h1) < EPS), 0)
    res += [t5]

    res = torch.stack(res, -1)

    # mask of all valid result (all non-negative results)
    posmask = res > -EPS
    # mask of all sdf-valid results (all results NOT outside of the cylinder)
    # the fully valid result is both non-negative and sdf-valid
    mask = posmask

    # we replace all non-valid results with +inf so that min only considers them when there are no valid results and therefore returns +inf as per specs
    minv = torch.min(res.masked_fill(~mask, torch.inf), -1)[0]
    # we replace all non-valid results with -inf so that max only considers them when there are no valid results and therefore returns -inf as per specs
    maxv = torch.max(res.masked_fill(~mask, -torch.inf), -1)[0]

    return minv, maxv


def test_cylinder_intersect():
    def intersect_and_check_sdf(o, d, r, h0, h1):
        o = np.array(o)
        d = np.array(d)

        res = cylinder_intersect(o, d, r, h0, h1)
        # print('t =', res)

        # 0 is excluded because it could be the special case when the origin is inside rather than directly on the surface
        # inf's are excluded because that's a special value for having no intersections
        pts = [o + d * np.array(t) for t in res if not np.allclose(t, 0) and not np.isinf(t)]
        # print('pts =', pts)
        # sdf_val = [cylinder_sdf_gold(p, r, h0, h1) for p in pts]
        sdf_val = [cylinder_sdf(p, r, h0, h1) for p in pts]
        # print('sdf_val =', sdf_val)
        assert_allclose(sdf_val, 0)
        return res

    # left-right with varying origin x
    assert_allclose(intersect_and_check_sdf((-2, 0, 0), (1, 0, 0), 1, -1, 1), [1, 3])
    assert_allclose(intersect_and_check_sdf((-1, 0, 0), (1, 0, 0), 1, -1, 1), [0, 2])
    assert_allclose(intersect_and_check_sdf((0,  0, 0), (1, 0, 0), 1, -1, 1), [0, 1])
    assert_allclose(intersect_and_check_sdf((1,  0, 0), (1, 0, 0), 1, -1, 1), [0, 0])
    assert_allclose(intersect_and_check_sdf((2,  0, 0), (1, 0, 0), 1, -1, 1), [torch.inf, -torch.inf])

    # top-down with varying origin y
    assert_allclose(intersect_and_check_sdf((0,  2, 0), (0, -1, 0), 1, -1, 1), [1, 3])
    assert_allclose(intersect_and_check_sdf((0,  1, 0), (0, -1, 0), 1, -1, 1), [0, 2])
    assert_allclose(intersect_and_check_sdf((0,  0, 0), (0, -1, 0), 1, -1, 1), [0, 1])
    assert_allclose(intersect_and_check_sdf((0, -1, 0), (0, -1, 0), 1, -1, 1), [0, 0])
    assert_allclose(intersect_and_check_sdf((0, -2, 0), (0, -1, 0), 1, -1, 1), [torch.inf, -torch.inf])

    # corner case (bottom plane)
    assert_allclose(intersect_and_check_sdf((-2, 0, 0), (1, 0, 0), 1, 0, 1), [1, 3])
    assert_allclose(intersect_and_check_sdf((-1, 0, 0), (1, 0, 0), 1, 0, 1), [0, 2])
    assert_allclose(intersect_and_check_sdf((0,  0, 0), (1, 0, 0), 1, 0, 1), [0, 1])
    assert_allclose(intersect_and_check_sdf((1,  0, 0), (1, 0, 0), 1, 0, 1), [0, 0])
    assert_allclose(intersect_and_check_sdf((2,  0, 0), (1, 0, 0), 1, 0, 1), [torch.inf, -torch.inf])

    # diagonal with both side and plane (w/collision)
    assert_allclose(intersect_and_check_sdf((0,   0, 0), (1, 1, 0), 1, 0, 1), [0, 1])
    assert_allclose(intersect_and_check_sdf((-1, -1, 0), (1, 1, 0), 1, 0, 1), [1, 2])

    # diagonal with both side and plane (w/o collision)
    assert_allclose(intersect_and_check_sdf((0,   0, 0), (1, 1, 0), 1, 0, 2), [0, 1])
    assert_allclose(intersect_and_check_sdf((-1, -1, 0), (1, 1, 0), 1, 0, 2), [1, 2])

    # top down with corner cases (side surface)
    assert_allclose(intersect_and_check_sdf((0, 1, 0), (0, -1, 0), 1, 0, 1), [0, 1])
    assert_allclose(intersect_and_check_sdf((1, 1, 0), (0, -1, 0), 1, 0, 1), [0, 1])
    # assert_allclose(intersect_and_check_sdf((2, 1, 0), (0, -1, 0), 1, 0, 1), [])
    assert_allclose(intersect_and_check_sdf((2, 1, 0), (0, -1, 0), 1, 0, 1), [torch.inf, -torch.inf])

    # degenerate case
    assert_allclose(intersect_and_check_sdf((0, 0, 0), (0, 0, 0), 1, -1, 1), [0, 0])

    # sample random rays from inside
    for _ in range(100):
        o = (0, 0, 0)
        d = np.random.random((3,))*1
        res = intersect_and_check_sdf(o, d, 1, -2, 3)
        assert len(res) == 2
        assert np.isclose(res[0], 0)

    # sample random rays from outside
    for _ in range(100):
        o = np.random.random((3,))*2
        d = np.random.random((3,))*1
        res = intersect_and_check_sdf(o, d, 1, -2, 3)
        assert len(res) == 2 or len(res) == 0


def test_cylinder_intersect_vs_gold():
    # sample random rays from inside
    for _ in range(100):
        o = (0, 0, 0)
        d = np.random.random((3,))*1
        res_gold = cylinder_intersect_gold(o, d, 1, -2, 3)
        res = cylinder_intersect(o, d, 1, -2, 3)
        assert_allclose(res_gold, res)

    # sample random rays from outside
    for _ in range(100):
        o = np.random.random((3,))*2
        d = np.random.random((3,))*1
        res_gold = cylinder_intersect_gold(o, d, 1, -2, 3)
        res = cylinder_intersect(o, d, 1, -2, 3)
        assert_allclose(res_gold, res)


def test_cylinder_intersect_batched():
    # --- first batch: 1, -1, 1
    os = torch.tensor([
        # left-right with varying origin x
        [-2, 0, 0],
        [-1, 0, 0],
        [0,  0, 0],
        [1,  0, 0],
        [2,  0, 0],

        # top-down with varying origin y
        [0,  2, 0],
        [0,  1, 0],
        [0,  0, 0],
        [0, -1, 0],
        [0, -2, 0],

        # degenerate case
        [0, 0, 0],
    ], dtype=torch.float64)


    ds = torch.tensor([
        # left-right with varying origin x
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        # top-down with varying origin y
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],

        # degenerate case
        [0, 0, 0],
    ], dtype=torch.float64)


    expected_min = torch.tensor([
        1, 0, 0, 0, torch.inf,
        1, 0, 0, 0, torch.inf,
        0,
    ], dtype=torch.float64)

    expected_max = torch.tensor([
        3, 2, 1, 0, -torch.inf,
        3, 2, 1, 0, -torch.inf,
        0,
    ], dtype=torch.float64)

    expected_res = (expected_min, expected_max)

    res = cylinder_intersect(os, ds, 1, -1, 1)

    assert_allclose(res, expected_res)

    del os, ds, expected_max, expected_min, res, expected_res

    # --- second batch: 1, 0, 1

    os = torch.tensor([
        # corner case (bottom plane)
        [-2, 0, 0],
        [-1, 0, 0],
        [0,  0, 0],
        [1,  0, 0],
        [2,  0, 0],

        # diagonal with both side and plane (w/collision)
        [0,   0, 0],
        [-1, -1, 0],

        # top down with corner cases (side surface)
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
    ], dtype=torch.float64)

    ds = torch.tensor([
        # corner case (bottom plane)
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        # diagonal with both side and plane (w/collision)
        [1, 1, 0],
        [1, 1, 0],

        # top down with corner cases (side surface)
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ], dtype=torch.float64)

    expected_min = torch.tensor([
        1, 0, 0, 0, torch.inf,
        0, 1,
        0, 0, torch.inf,
    ], dtype=torch.float64)

    expected_max = torch.tensor([
        3, 2, 1, 0, -torch.inf,
        1, 2,
        1, 1, -torch.inf,
    ], dtype=torch.float64)

    expected_res = (expected_min, expected_max)

    res = cylinder_intersect(os, ds, 1, 0, 1)

    assert_allclose(res, expected_res)

    del os, ds, expected_max, expected_min, res, expected_res

    # --- third batch: 1, 0, 2

    os = torch.tensor([
        # diagonal with both side and plane (w/o collision)
        [0,   0, 0],
        [-1, -1, 0],

    ], dtype=torch.float64)

    ds = torch.tensor([
        # diagonal with both side and plane (w/o collision)
        [1, 1, 0],
        [1, 1, 0],
    ], dtype=torch.float64)

    expected_min = torch.tensor([
        0, 1
    ], dtype=torch.float64)

    expected_max = torch.tensor([
        1, 2
    ], dtype=torch.float64)

    expected_res = (expected_min, expected_max)

    res = cylinder_intersect(os, ds, 1, 0, 2)

    assert_allclose(res, expected_res)


def test_cylinder_intersect_different_shapes():
    os = torch.tensor([
        # diagonal with both side and plane (w/o collision)
        [0,   0, 0],
        [-1, -1, 0],

        [1,   1, 0],
        [2,   2, 0],

    ], dtype=torch.float64)

    os = os.view(2, 1, 2, 3)


    ds = torch.tensor([
        # diagonal with both side and plane (w/o collision)
        [1, 1, 0],
        [1, 1, 0],

        [1, 1, 0],
        [1, 1, 0],
    ], dtype=torch.float64)

    ds = ds.view(2, 1, 2, 3)

    expected_min = torch.tensor([
        [[0, 1]],
        [[0, torch.inf]]
    ], dtype=torch.float64)

    expected_max = torch.tensor([
        [[1, 2]],
        [[0, -torch.inf]],
    ], dtype=torch.float64)

    expected_res = (expected_min, expected_max)

    res = cylinder_intersect(os, ds, 1, 0, 2)

    assert res[0].shape == os.shape[:-1]
    assert res[1].shape == os.shape[:-1]

    assert res[0].shape == expected_min.shape
    assert res[1].shape == expected_max.shape

    assert_allclose(res, expected_res)


def test_cylinder_intersect_precision():
    TYPE = torch.float32
    # sample random rays from inside
    origin_offset = torch.tensor([3, 0, 0], dtype=TYPE)
    BS = 100
    d = torch.tensor([[-1, 0, 0] for _ in range(BS)], dtype=TYPE)
    for _ in range(10000):
        # so that it is at least -1+3=2>1
        # o = torch.rand(3, dtype=TYPE)*1 + origin_offset
        o = torch.rand((BS, 3), dtype=TYPE)*1 + origin_offset

        # res_gold = cylinder_intersect_gold(o, d, 1, -2, 2)
        # breakpoint()

        # cast to float32
        res = cylinder_intersect(o, d, r=1, h0=-2, h1=2)
        # assert_allclose(res_gold, res)

        # make sure that both of the intersections exist (if they don't exist, the returned values are min=+inf max=-inf)
        try:
            assert not torch.any(torch.isinf(res[0]))
            assert not torch.any(torch.isinf(res[1]))
        except AssertionError:
            print(o, d, res)
            raise


def test_cylinder_intersect_precision_cuda():
    TYPE = torch.float16
    # sample random rays from inside
    origin_offset = torch.tensor([3, 0, 0], dtype=TYPE).cuda()
    BS = 100
    d = torch.tensor([[-1, 0, 0] for _ in range(BS)], dtype=TYPE).cuda()
    for _ in range(10000):
        # so that it is at least -1+3=2>1
        # o = torch.rand(3, dtype=TYPE)*1 + origin_offset
        o = torch.rand((BS, 3), dtype=TYPE).cuda()*1 + origin_offset

        # res_gold = cylinder_intersect_gold(o, d, 1, -2, 2)
        # breakpoint()

        # cast to float32
        res = cylinder_intersect(o, d, r=1, h0=-2, h1=2)
        # assert_allclose(res_gold, res)

        # make sure that both of the intersections exist (if they don't exist, the returned values are min=+inf max=-inf)
        try:
            assert not torch.any(torch.isinf(res[0]))
            assert not torch.any(torch.isinf(res[1]))
        except AssertionError:
            print(o, d, res)
            # breakpoint()
            cylinder_intersect(o, d, r=1, h0=-2, h1=2)

            raise


if __name__ == '__main__':
    # import pytest
    # pytest.main([__file__])

    test_cylinder_sdf()
    test_cylinder_sdf_vs_gold()
    test_cylinder_intersect()
    test_cylinder_intersect_vs_gold()
    #test_cylinder_intersect_batched()
    #test_cylinder_intersect_different_shapes()
    test_cylinder_intersect_precision()
    test_cylinder_intersect_precision_cuda()
