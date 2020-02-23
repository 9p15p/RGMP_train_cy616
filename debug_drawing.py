#dubug时候画图用的代码
import matplotlib.pyplot as plt
import numpy as np
def debug_draw(all_E):
    debug_show_pic = all_E
    try:
        print("the picture shape:",debug_show_pic.squeeze().shape)
    except:
        print("the picture shape:",debug_show_pic.squeeze().size)

    try:
        plt.imshow(np.transpose(debug_show_pic.squeeze(),[1,2,0]))
    except:
        plt.imshow(debug_show_pic.squeeze())
    plt.show()
