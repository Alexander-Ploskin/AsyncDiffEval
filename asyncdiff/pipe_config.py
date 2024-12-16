def splite_model(pipe, n):
    unet = pipe.unet

    if n == 2:
        return [
            (
            unet.conv_in,
            *unet.down_blocks,
            unet.mid_block,
            unet.up_blocks[0],
            unet.up_blocks[1],
        ),
            (
            *unet.up_blocks[2:],
            unet.conv_norm_out,
            unet.conv_out
        )
        ]
    elif n == 3:
        return [
            (
            unet.conv_in,
            unet.down_blocks[0],
            unet.down_blocks[1],
            unet.down_blocks[2],
        ),
            (
            unet.down_blocks[3],
            unet.mid_block,
            unet.up_blocks[0],
            unet.up_blocks[1],
            unet.up_blocks[2],
        ),
            (
            unet.up_blocks[3],
            unet.conv_norm_out,
            unet.conv_out
        )
        ]
    elif n == 4:
        return [
            (
            unet.down_blocks[1].resnets[0],
            unet.down_blocks[1].attentions[0],
            unet.conv_in,
            unet.down_blocks[0],
        ),
            (
            unet.down_blocks[1].resnets[1],
            unet.down_blocks[1].attentions[1],
            *unet.down_blocks[1].downsamplers,
            *unet.down_blocks[2:4],
            unet.mid_block,
            unet.up_blocks[0],
            unet.up_blocks[1],
        ),
            (
            unet.up_blocks[2],
            unet.up_blocks[3].resnets[0],
            unet.up_blocks[3].attentions[0],
        ),
        (
            unet.up_blocks[3].resnets[1],
            unet.up_blocks[3].attentions[1],
            unet.up_blocks[3].resnets[2],
            unet.up_blocks[3].attentions[2],
            unet.conv_norm_out,
            unet.conv_out
        )
        ]
    else:
        raise NotImplementedError
