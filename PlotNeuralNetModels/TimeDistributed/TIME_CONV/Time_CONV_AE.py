
import sys
sys.path.append('../../../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '../../..' ),
    to_cor(),
    to_begin(),

    # input
    to_input('../../../pyexamples/mspacman.png'),

    to_Conv("conv1", s_filer=246480, n_filer=10, offset="(0,0,0)", to="(0,0,0)", height=35, depth=30, width=2 , caption="TimeDistConv2D"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=3, depth=3),

    to_Conv("conv2", 7956, 10, offset="(1.5,0,0)", to="(pool1-east)", height=15, depth=10, width=2 ,caption="TimeDistConv2D"),
    to_connection( "pool1", "conv2"),

    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=3, depth=3, width=1),

    to_SoftMax(name="flat1", s_filer= 8400 ,offset= "(2.0,0,0)", to = "(pool2-east)", caption="Flatten",
               height=20, depth=20, width=1),
    to_connection("pool2", "flat1"),

    to_Conv(name="dense1", s_filer=2386830, n_filer=30, offset= "(2.0,0,0)", to="(flat1-east)", caption="dense1",
            height=19, depth=19, width=2),
    to_connection("flat1", "dense1"),
    
    to_SoftMax(name="inference_outputs", s_filer=30, offset="(1,0,0)", to="(dense1-east)", caption="Latents",
            height=5, depth=5, width=10),
    to_connection("dense1", "inference_outputs"),
    
    
    

    # Decoder
    to_Conv(name='dense2', s_filer=210000, n_filer=30, offset="(1,0,0)", to="(inference_outputs-east)",
            width=5, height=10, depth=10, caption="dense2"),
    to_connection("inference_outputs", "dense2"),

    to_Conv(name='reshape', s_filer=21000, n_filer=15, offset="(1,0,0)", to="(dense2-east)",
            height=15, depth=13, width=2, caption="Reshape"),
    to_connection("dense2", "reshape"),

    to_Conv(name='deconv3', s_filer=8400, n_filer=300, offset="(1.5,0,0)", to="(reshape-east)",
            height=15, depth=13, width=4, caption="TimeDistConv2DTr1"),
    to_Pool("pool4", offset="(0,0,0)", to="(deconv3-east)", height=2, depth=3, width=1),
    to_connection("reshape", "deconv3"),

    to_Conv(name='deconv4', s_filer=160*210, n_filer=60, offset="(2,0,0)", to="(deconv3-east)",
            height=35, depth=30, width=2, caption="TimeDistConv2DTr2"),
    to_Pool("pool5", offset="(0,0,0)", to="(deconv4-east)", height=2, depth=2, width=1),
    to_connection("pool4", "deconv4"),

    to_Conv(name='deconv5', s_filer=10*160*210, n_filer=3, offset="(5,0,0)", to="(deconv3-east)",
            height=45, depth=40, width=2, caption="TimeDistConv2DTr3"),
    to_connection("pool5", "deconv5"),
    
    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.45,0,0)", to="(deconv5-east)",
                   width=1, height=45, depth=40, ),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
