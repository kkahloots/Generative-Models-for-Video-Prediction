
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

    to_input('../../../../pyexamples/normaldist.png',to='(-3,-10,0)', width=10, height=10, name="NormalDist"),



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
    #
    # #inferense logvar
    # to_Conv("convb1", s_filer=100800, n_filer=300, offset="(0,-10,0)", to="(0,0,0)", height=45, depth=40, width=2 , caption="conv1"),
    # #the hight and weight of the pool layer is depending on kernel_size=3
    # to_Pool("poolb1", offset="(0,0,0)", to="(convb1-east)", height=3, depth=3),
    #
    # #since my stride is (2,2) that means my dimentions will be decreased by 4 times but my intermediate dim is 30/2 so 15/3= 5
    # # thus i have to multiply n_filer by 5
    # to_Conv("convb2", 25200, 1500, offset="(1.5,0,0)", to="(poolb1-east)", height=15, depth=13, width=2 ,caption="conv2"),
    # to_connection( "poolb1", "convb2"),
    #
    # to_Pool("poolb2", offset="(0,0,0)", to="(convb2-east)", height=3, depth=3, width=1),
    #
    # to_SoftMax(name="flatb1", s_filer= 8400 ,offset= "(2.0,0,0)", to = "(poolb2-east)", caption="Flatten",
    #            height=20, depth=20, width=1),
    # # to_Pool(name='Flatten', to),
    #
    # to_connection("poolb2", "flatb1"),
    #
    # to_Conv(name="denseb1", s_filer=238710, n_filer=30, offset= "(2.5,0,0)", to="(flatb1-east)", caption="dense1",
    #         height=19, depth=19, width=2),
    # to_connection("flatb1", "denseb1"),
    #
    # to_SoftMax(name="inference_log_outputs", s_filer=30, offset="(1,0,0)", to="(dense1-east)", caption="Latents",
    #            height=5, depth=5, width=10),
    # to_connection("denseb1", "inference_log_outputs"),
    

    #generator
    to_Conv("genconv1", s_filer=100800, n_filer=300, offset="(0,0,0)", to="(0,-10,0)", height=45, depth=40, width=2 , caption="CONV"),
    to_Pool("genpool1", offset="(0,0,0)", to="(genconv1-east)", height=3, depth=3),


    to_Conv("genconv2", 25200, 1500, offset="(1.5,0,0)", to="(genpool1-east)", height=15, depth=13, width=2 ,caption="CONV"),
    to_connection( "genpool1", "genconv2"),

    to_Pool("genpool2", offset="(0,0,0)", to="(genconv2-east)", height=3, depth=3, width=1),

    to_SoftMax(name="genflat1", s_filer= 8400 ,offset= "(2.0,0,0)", to = "(genpool2-east)", caption="Flatten",
               height=20, depth=20, width=1),
    to_connection("genpool2", "genflat1"),

    to_Conv(name="gendense1", s_filer=238710, n_filer=30, offset= "(2.5,0,0)", to="(genflat1-east)", caption="DENSE",
            height=19, depth=19, width=2),
    to_connection("genflat1", "gendense1"),

    to_SoftMax(name="genflat2", s_filer= 30 ,offset= "(2.0,0,0)", to = "(gendense1-east)", caption="Flatten",
               height=5, depth=5, width=1),
    to_connection("gendense1", "genflat2"),

    to_SoftMax(name="gen_outputs", s_filer=30, offset="(1,0,0)", to="(genflat2-east)", caption="FakeLatents",
               height=5, depth=5, width=10),
    to_connection("genflat2", "gen_outputs"),

    #fake value discriminator
    to_Conv("discdense01", s_filer=160*210, n_filer=15, offset="(1.0,2,0)", to="(gen_outputs-east)", height=15, depth=13, width=5,
            caption="DENSE"),
        to_connection("gen_outputs", "discdense01"),
        to_connection("inference_outputs", "discdense01"),
    to_Conv("discdense02", 160*210, 15, offset="(1.0,0,0)", to="(discdense01-east)", height=15, depth=13, width=5,
            caption="DENSE"),
    to_connection( "discdense01", "discdense02"),

    to_SoftMax(name="discflat1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(discdense02-east)", caption="Flatten",
               height=20, depth=20, width=1),

    to_connection("discdense02", "discflat1"),

    to_Conv(name="discdense03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(discflat1-east)", caption="DENSE",
            height=5, depth=4, width=15),
    to_connection("discflat1", "discdense03"),

    to_SoftMax(name="discflat2", s_filer= 30 ,offset= "(1.0,0,0)", to = "(discdense03-east)", caption="Flatten",
               height=5, depth=5, width=1),
    to_connection("discdense03", "discflat2"),

    to_SoftMax(name="disc_outputs", s_filer=1, offset="(1,0,0)", to="(discflat2-east)", caption="Binary",
               height=1, depth=1, width=1),
    to_connection("discflat2", "disc_outputs"),


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
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
