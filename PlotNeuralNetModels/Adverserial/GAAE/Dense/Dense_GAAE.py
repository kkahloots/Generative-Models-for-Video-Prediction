
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

    # inference mean
    to_Conv("dense01", s_filer=160*210, n_filer=15, offset="(0,0,0)", to="(0,0,0)", height=15, depth=13, width=5,
            caption="dense01"),

    to_Conv("dense02", 160*210, 15, offset="(1.0,0,0)", to="(dense01-east)", height=15, depth=13, width=5,
            caption="dense02"),
    to_connection( "dense01", "dense02"),

    to_SoftMax(name="flat1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(dense02-east)", caption="Flatten",
               height=20, depth=20, width=1),

    to_connection("dense02", "flat1"),

    to_Conv(name="dense03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(flat1-east)", caption="dense03",
            height=5, depth=4, width=15),
    to_connection("flat1", "dense03"),

    to_SoftMax(name="inference_outputs", s_filer=30, offset="(1,0,0)", to="(dense03-east)", caption="Latents",
               height=5, depth=5, width=10),
    to_connection("dense03", "inference_outputs"),


#     #real value discriminator
#     to_Conv("discrealdense01", s_filer=160*210, n_filer=15, offset="(0,0,0)", to="(10,-10,0)", height=15, depth=13, width=5,
#             caption="discrealdense01"),

#     to_Conv("discrealdense02", 160*210, 15, offset="(1.0,0,0)", to="(discrealdense01-east)", height=15, depth=13, width=5,
#             caption="discrealdense02"),
#     to_connection( "discrealdense01", "discrealdense02"),

#     to_SoftMax(name="discrealflat1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(discrealdense02-east)", caption="Flatten",
#                height=20, depth=20, width=1),

#     to_connection("discrealdense02", "discrealflat1"),

#     to_Conv(name="discrealdense03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(discrealflat1-east)", caption="dense03",
#             height=5, depth=4, width=15),
#     to_connection("discrealflat1", "discrealdense03"),

#     to_SoftMax(name="discrealflat2", s_filer= 30 ,offset= "(1.0,0,0)", to = "(discrealdense03-east)", caption="Flatten",
#                height=5, depth=5, width=1),
#     to_connection("discrealdense03", "discrealflat2"),

#     to_SoftMax(name="discreal_outputs", s_filer=1, offset="(1,0,0)", to="(discrealdense03-east)", caption="Binary",
#                height=1, depth=1, width=1),
#     to_connection("discrealdense03", "discreal_outputs"),


    # generator
    to_Conv("gendense01", s_filer=160*210, n_filer=15, offset="(0,0,0)", to="(0,-10,0)", height=15, depth=13, width=5,
            caption="DENSE"),

    to_Conv("gendense02", 160*210, 15, offset="(1.0,0,0)", to="(gendense01-east)", height=15, depth=13, width=5,
            caption="DENSE"),
    to_connection( "gendense01", "gendense02"),

    to_SoftMax(name="genflat1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(gendense02-east)", caption="Flatten",
               height=20, depth=20, width=1),
    to_connection("gendense02", "genflat1"),

    to_Conv(name="gendense03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(genflat1-east)", caption="DENSE",
            height=5, depth=4, width=15),
    to_connection("genflat1", "gendense03"),

    to_SoftMax(name="genflat2", s_filer= 30 ,offset= "(1.0,0,0)", to = "(gendense03-east)", caption="Flatten",
               height=5, depth=5, width=1),
    to_connection("gendense03", "genflat2"),

    to_SoftMax(name="gen_outputs", s_filer=30, offset="(1,0,0)", to="(genflat2-east)", caption="FakeLatents",
               height=5, depth=5, width=10),
    to_connection("genflat2", "gen_outputs"),


            #fake value discriminator
    to_Conv("discdense01", s_filer=160*210, n_filer=15, offset="(1.5,2,0)", to="(gen_outputs-east)", height=15, depth=13, width=5,
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
    to_Conv(name='dense04', s_filer=30, n_filer=504000, offset="(1,0,0)", to="(inference_outputs-east)",
            height=5, depth=4, width=15, caption="dense04"),
    to_connection("inference_outputs", "dense04"),


    to_Conv(name='dense05', s_filer=160*210, n_filer=15, offset="(1.5,0,0)", to="(dense04-east)",
            height=15, depth=13, width=5, caption="dense05"),
    to_connection("dense04", "dense05"),

    to_Conv(name='dense06', s_filer=160*210, n_filer=15, offset="(2,0,0)", to="(dense05-east)",
            height=15, depth=13, width=5, caption="dense06"),
    to_connection("dense05", "dense06"),

    to_Conv(name='reshape', s_filer=160*210, n_filer=3, offset="(1,0,0)", to="(dense06-east)",
            height=45, depth=40, width=2, caption="Reshape"),
    to_connection("dense06", "reshape"),

    to_ConvSoftMax(name="soft1", s_filer=512, offset="(0.45,0,0)", to="(reshape-east)",
                   width=1, height=45, depth=40, ),
    
    

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
