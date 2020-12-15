
import sys
sys.path.append('../../../../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '../../../..' ),
    to_cor(),
    to_begin(),

    # input
    to_input('../../../../pyexamples/mspacman.png'),

    to_input('../../../../pyexamples/normaldist.png',to='(9,-10,0)', width=7, height=5, name="NormalDist"),



    to_Conv("conv1", s_filer=100800, n_filer=300, offset="(0,0,0)", to="(0,0,0)", height=45, depth=40, width=2 , caption="conv1"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=3, depth=3),


    to_Conv("conv2", 25200, 1500, offset="(1.5,0,0)", to="(pool1-east)", height=15, depth=13, width=2 ,caption="conv2"),
    to_connection( "pool1", "conv2"),

    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=3, depth=3, width=1),

    to_SoftMax(name="flat1", s_filer= 8400 ,offset= "(2.0,0,0)", to = "(pool2-east)", caption="Flatten",
               height=20, depth=20, width=1),
    to_connection("pool2", "flat1"),

    to_Conv(name="dense1", s_filer=238710, n_filer=30, offset= "(2.5,0,0)", to="(flat1-east)", caption="dense1",
            height=19, depth=19, width=2),
    to_connection("flat1", "dense1"),

    to_SoftMax(name="inference_outputs", s_filer=30, offset="(1,0,0)", to="(dense1-east)", caption="Latents",
               height=5, depth=5, width=10),
    to_connection("dense1", "inference_outputs"),
 
    # Decoder
    to_Conv(name='dense2', s_filer=21000, n_filer=30, offset="(1,0,0)", to="(inference_outputs-east)",
            width=5, height=10, depth=10, caption="dense2"),
    to_connection("inference_outputs", "dense2"),

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

    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.45,0,0)", to="(deconv5-east)",
                   width=1, height=45, depth=40, ),

    

    #infgenerator
    to_Conv("infgendense", s_filer=21000, n_filer=30, offset="(0,0,0)", to="(11.5,-11,0)", width=5, height=10, depth=10 
    ,caption="DENSE"),

    to_Conv("infgenreshape", s_filer=14000, n_filer=15, offset="(1.5,0,0)", to="(infgendense-east)", height=15, depth=13, width=2 
    ,caption="Reshape"),
    to_connection( "infgendense", "infgenreshape"),

    to_Conv(name='infgendeconv3', s_filer=84000, n_filer=6, offset="(1.5,0,0)", to="(infgenreshape-east)",
            height=15, depth=13, width=2, caption="conv2dT"),
    to_Pool("infgenpool4", offset="(0,0,0)", to="(infgendeconv3-east)", height=2, depth=3, width=1),
    to_connection("infgenreshape", "infgendeconv3"),

    to_Conv(name='infgendeconv4', s_filer=336000, n_filer=30, offset="(2,0,0)", to="(infgendeconv3-east)",
            height=45, depth=40, width=2, caption="conv2dT"),
    to_Pool("infgenpool5", offset="(0,0,0)", to="(infgendeconv4-east)", height=2, depth=2, width=1),
    to_connection("infgenpool4", "infgendeconv4"),

    to_Conv(name='infgendeconv5', s_filer=336000, n_filer=3, offset="(5,0,0)", to="(infgendeconv3-east)",
            height=45, depth=40, width=2, caption="conv2dT"),
    to_connection("infgenpool5", "infgendeconv5"),

    to_SoftMax(name="infgen_outputs", s_filer=30, offset="(1,0,0)", to="(infgendeconv5-east)", caption="FakeIMG",
               height=5, depth=5, width=10),
    to_connection("infgendeconv5", "infgen_outputs"),

    #fake value infdiscriminator
    to_Conv("infdiscdense01", s_filer=160*210, n_filer=15, offset="(2.5,5.5,0)", to="(infgen_outputs-east)", height=15, depth=13, width=5,
            caption="DENSE"),
        to_connection("infgen_outputs", "infdiscdense01"),
        to_connection("soft1", "infdiscdense01"),
    to_Conv("infdiscdense02", 160*210, 15, offset="(1.0,0,0)", to="(infdiscdense01-east)", height=15, depth=13, width=5,
            caption="DENSE"),
    to_connection( "infdiscdense01", "infdiscdense02"),

    to_SoftMax(name="infdiscflat1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(infdiscdense02-east)", caption="Flatten",
               height=20, depth=20, width=1),

    to_connection("infdiscdense02", "infdiscflat1"),

    to_Conv(name="infdiscdense03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(infdiscflat1-east)", caption="DENSE",
            height=5, depth=4, width=15),
    to_connection("infdiscflat1", "infdiscdense03"),

    to_SoftMax(name="infdiscflat2", s_filer= 30 ,offset= "(1.0,0,0)", to = "(infdiscdense03-east)", caption="Flatten",
               height=5, depth=5, width=1),
    to_connection("infdiscdense03", "infdiscflat2"),

    to_SoftMax(name="infdisc_outputs", s_filer=1, offset="(1,0,0)", to="(infdiscflat2-east)", caption="Binary",
               height=1, depth=1, width=1),
    to_connection("infdiscflat2", "infdisc_outputs"),


    


    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
