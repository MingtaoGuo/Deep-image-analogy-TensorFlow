import numpy as np
import cv2


def normalize(F_L):
    return F_L/np.sqrt(np.sum(np.square(F_L)))

def initialise_nnf(A, AA, B, BB, nnf, patch_size, is_init):
    """
    Set up a random NNF
    Then calculate the distances to fill up the NND
    :return:
    """
    A = normalize(A)
    AA = normalize(AA)
    B = normalize(B)
    BB = normalize(BB)
    nnd = np.zeros(shape=(A.shape[0], A.shape[1]))  # the distance map for the nnf
    if not is_init:
        nnf = np.zeros(shape=(2, A.shape[0], A.shape[1])).astype(np.int)  # the nearest neighbour field
        nnf[0] = np.random.randint(B.shape[1], size=(A.shape[0], A.shape[1]))
        nnf[1] = np.random.randint(B.shape[0], size=(A.shape[0], A.shape[1]))
        nnf = nnf.transpose((1, 2, 0))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            pos = nnf[i, j]
            nnd[i, j] = cal_dist(A, AA, B, BB, i, j, pos[1], pos[0], patch_size)
    return nnf, nnd

def cal_dist(A, AA, B, BB, ay, ax, by, bx, patch_size):
    """
    Calculate distance between a patch in A to a patch in B.
    :return: Distance calculated between the two patches
    """
    dx0 = dy0 = patch_size // 2
    dx1 = dy1 = patch_size // 2 + 1
    dx0 = min(ax, bx, dx0)
    dx1 = min(A.shape[0] - ax, B.shape[0] - bx, dx1)
    dy0 = min(ay, by, dy0)
    dy1 = min(A.shape[1] - ay, B.shape[1] - by, dy1)
    return np.sum(((A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - B[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2) +
                  ((AA[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - BB[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2)) / ( (dx1 + dx0) * (dy1 + dy0))

def reconstruct_image(img_a, nnf):
    """
    Reconstruct image using the NNF and img_a.
    :param img_a: the patches to reconstruct from
    :return: reconstructed image
    """
    final_img = np.zeros_like(img_a)
    size = nnf.shape[0]
    scale = img_a.shape[0] // nnf.shape[0]
    for i in range(size):
        for j in range(size):
            x, y = nnf[i, j]
            if final_img[scale * i:scale * (i + 1), scale * j:scale * (j + 1)].shape == img_a[scale * y:scale * (y + 1),
                                                                                        scale * x:scale * (x + 1)].shape:
                final_img[scale * i:scale * (i + 1), scale * j:scale * (j + 1)] = img_a[scale * y:scale * (y + 1),
                                                                                  scale * x:scale * (x + 1)]
    return final_img


def reconstruct_avg(img, nnf, patch_size=5):
    """
    Reconstruct image using average voting.
    :param img: the image to reconstruct from. Numpy array of dim H*W*3
    :param patch_size: the patch size to use

    :return: reconstructed image
    """

    final = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            dx0 = dy0 = patch_size // 2
            dx1 = dy1 = patch_size // 2 + 1
            dx0 = min(j, dx0)
            dx1 = min(img.shape[0] - j, dx1)
            dy0 = min(i, dy0)
            dy1 = min(img.shape[1] - i, dy1)

            patch = nnf[i - dy0:i + dy1, j - dx0:j + dx1]

            lookups = np.zeros(shape=(patch.shape[0], patch.shape[1], img.shape[2]), dtype=np.float32)

            for ay in range(patch.shape[0]):
                for ax in range(patch.shape[1]):
                    x, y = patch[ay, ax]
                    lookups[ay, ax] = img[y, x]

            if lookups.size > 0:
                value = np.mean(lookups, axis=(0, 1))
                final[i, j] = value

    return final

def upsample_nnf(nnf):
    """
    Upsample NNF based on size. It uses nearest neighbour interpolation
    :param size: INT size to upsample to.

    :return: upsampled NNF
    """

    temp = np.zeros((nnf.shape[0], nnf.shape[1], 3))

    for y in range(nnf.shape[0]):
        for x in range(nnf.shape[1]):
            temp[y][x] = [nnf[y][x][0], nnf[y][x][1], 0]

    # img = np.zeros(shape=(size, size, 2), dtype=np.int)
    # small_size = nnf.shape[0]
    aw_ratio = 2#((size) // small_size)
    ah_ratio = 2#((size) // small_size)

    temp = cv2.resize(temp, None, fx=aw_ratio, fy=aw_ratio, interpolation=cv2.INTER_NEAREST)
    img = np.zeros(shape=(temp.shape[0], temp.shape[1], 2), dtype=np.int)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            pos = temp[i, j]
            img[i, j] = pos[0] * aw_ratio, pos[1] * ah_ratio

    return img

def propagate(A, AA, B, BB, nnf, nnd, iters=2, rand_search_radius=6, patch_size=3, queue=None):
    """
    Optimize the NNF using PatchMatch Algorithm
    :param iters: number of iterations
    :param rand_search_radius: max radius to use in random search
    :return:
    """
    A = normalize(A)
    AA = normalize(AA)
    B = normalize(B)
    BB = normalize(BB)
    a_cols = A.shape[1]
    a_rows = A.shape[0]

    b_cols = B.shape[1]
    b_rows = B.shape[0]
    print("NNF Searching...")
    print("|", end="")
    for it in range(iters):
        ystart = 0
        yend = a_rows
        ychange = 1
        xstart = 0
        xend = a_cols
        xchange = 1

        if it % 2 == 1:
            xstart = xend - 1
            xend = -1
            xchange = -1
            ystart = yend - 1
            yend = -1
            ychange = -1

        ay = ystart
        while ay != yend:

            ax = xstart
            while ax != xend:

                xbest, ybest = nnf[ay, ax]
                dbest = nnd[ay, ax]
                if ax - xchange < a_cols and ax - xchange >= 0:
                    vp = nnf[ay, ax - xchange]
                    xp = vp[0] + xchange
                    yp = vp[1]
                    if xp < b_cols and xp >= 0:
                        val = cal_dist(A, AA, B, BB, ay, ax, yp, xp, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = xp, yp, val

                if abs(ay - ychange) < a_rows and ay - ychange >= 0:
                    vp = nnf[ay - ychange, ax]
                    xp = vp[0]
                    yp = vp[1] + ychange
                    if yp < b_rows and yp >= 0:
                        val = cal_dist(A, AA, B, BB, ay, ax, yp, xp, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = xp, yp, val
                if rand_search_radius is None:
                    rand_d = max(B.shape[0], B.shape[1])
                else:
                    rand_d = rand_search_radius

                while rand_d >= 1:
                    try:
                        xmin = max(xbest - rand_d, 0)
                        xmax = min(xbest + rand_d, b_cols)

                        ymin = max(ybest - rand_d, 0)
                        ymax = min(ybest + rand_d, b_rows)

                        if xmin > xmax:
                            rx = -np.random.randint(xmax, xmin)
                        if ymin > ymax:
                            ry = -np.random.randint(ymax, ymin)

                        if xmin <= xmax:
                            rx = np.random.randint(xmin, xmax)
                        if ymin <= ymax:
                            ry = np.random.randint(ymin, ymax)

                        val = cal_dist(A, AA, B, BB, ay, ax, ry, rx, patch_size)
                        if val < dbest:
                            xbest, ybest, dbest = rx, ry, val

                    except Exception as e:
                        print(e)
                        print(rand_d)
                        print(xmin, xmax)
                        print(ymin, ymax)
                        print(xbest, ybest)
                        print(B.shape)

                    rand_d = rand_d // 2

                nnf[ay, ax] = [xbest, ybest]
                nnd[ay, ax] = dbest

                ax += xchange
            ay += ychange
        print("{}".format("->"+str(it+1)), end="")
    print("|")
    if queue:
        queue.put(nnf)
    return nnf, nnd