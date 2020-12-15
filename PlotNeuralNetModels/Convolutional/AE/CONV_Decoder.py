
import sys
sys.path.append('../')
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # Decoder
    to_Conv(name='dense2', s_filer=21000, n_filer=30, offset="(0,0,0)", to="(0,0,0)",
                    width=5, height=10, depth=10, caption="dense2"),

    to_Conv(name='reshape', s_filer=14000, n_filer=15, offset="(1,0,0)", to="(dense2-east)",
                    height=15, depth=13, width=2, caption="Reshape"),
    to_connection("dense2", "reshape"),

    to_Conv(name='deconv3', s_filer=84000, n_filer=6, offset="(1.5,0,0)", to="(reshape-east)",
                    height=15, depth=13, width=2, caption="conv2dT1"),
    to_Pool("pool4", offset="(0,0,0)", to="(deconv3-east)", height=2, depth=3, width=1),
    to_connection("reshape", "deconv3"),

    to_Conv(name='deconv4', s_filer=336000, n_filer=30, offset="(2,0,0)", to="(deconv3-east)",
                    height=45, depth=40, width=2, caption="conv2dT2"),
    to_Pool("pool5", offset="(0,0,0)", to="(deconv4-east)", height=2, depth=2, width=1),
    to_connection("pool4", "deconv4"),

    to_Conv(name='deconv5', s_filer=336000, n_filer=3, offset="(5,0,0)", to="(deconv3-east)",
                    height=45, depth=40, width=2, caption="conv2dT3"),
    to_connection("pool5", "deconv5"),

    to_ConvSoftMax( name="soft1", s_filer=512, offset="(0.45,0,0)", to="(deconv5-east)",
                    width=1, height=45, depth=40,),
    # to_connection( "deconv5", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
