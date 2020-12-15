
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # input
    to_input('../pyexamples/mspacman.png'),

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

    to_SoftMax(name="inference_outputs", s_filer=30, offset="(1.5,-2.5,0)", to="(dense03-east)", caption="Latents",
               height=5, depth=5, width=10),
    to_connection("dense03", "inference_outputs"),

    # inference logvar
    to_Conv("denseb01", s_filer=160*210, n_filer=15, offset="(0,-5,0)", to="(0,0,0)", height=15, depth=13, width=5,
            caption="denseb01"),

    to_Conv("denseb02", 160*210, 15, offset="(1.0,0,0)", to="(denseb01-east)", height=15, depth=13, width=5,
            caption="denseb02"),
    to_connection( "denseb01", "denseb02"),

    to_SoftMax(name="flatb1", s_filer= 504000 ,offset= "(1.0,0,0)", to = "(denseb02-east)", caption="Flatten",
               height=20, depth=20, width=1),

    to_connection("denseb02", "flatb1"),

    to_Conv(name="denseb03", s_filer=30, n_filer=504000, offset= "(1.0,0,0)", to="(flatb1-east)", caption="denseb03",
            height=5, depth=4, width=15),
    to_connection("flatb1", "denseb03"),

    to_connection("denseb03", "inference_outputs"),

    # Decoder
    to_Conv(name='dense04', s_filer=30, n_filer=504000, offset="(1,0,0)", to="(inference_outputs-east)",
            height=5, depth=4, width=15, caption="dense04"),
    to_connection("inference_outputs", "dense04"),


    to_Conv(name='dense05', s_filer=160*210, n_filer=15, offset="(1.5,0,0)", to="(dense04-east)",
            height=15, depth=13, width=5, caption="dense05"),
    to_connection("dense04", "dense05"),

    to_Conv(name='dense06', s_filer=160*210, n_filer=15, offset="(1.0,0,0)", to="(dense05-east)",
            height=15, depth=13, width=5, caption="dense06"),
    to_connection("dense05", "dense06"),

    to_Conv(name='reshape', s_filer=160*210, n_filer=3, offset="(2,0,0)", to="(dense06-east)",
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
