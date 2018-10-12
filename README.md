        # Examples:
        #    1
        # 1  2
        # 2  3
        #    4

        # 1 <- w * 1
        # 2 <- w * 1 + w * 2
        # 3 <- w * 1 + x * 2
        # 4 <- w * 2

        #
        #    1
        # 1  2
        # 2  3
        # 3  4
        #    5
        #

        # 1 <- w * 1
        # 2 >- w * 1 + w * 2
        # 3 <- w * 1 + w * 2 + w * 3
        # 4 <- w * 2 + w * 3
        # 5 <- w * 3

        # Pad in by 1, out by 2. roll i:i + 3
        # W needs dim to match
        # *  *    
        # *  1
        # 1  2
        # 2  3
        # *  4
        # *  *

        
        # *  x  w1  -> sum -> 1
        # *  x  w2
        # 1  x  w3

        #           [*, *] -> sum -> 1
        #           [*, *]
        # [0, 1]  x [w, w]


        # Shrink
        # 1
        # 2  1
        # 3  2
        # 4

        # 1 <- 1 * w + 2 * w + 3 * w
        # 2 <- 2 * w + 3 * w + 4 * w

        # Works as is