class SNNMultiShortcutBlock(nn.Module): 、
    def forward_time(self, x, mem, spike, BiSNN):
        residual = x
        out = self.binary_activation(x)
        out1 = self.bn1(self.binary_conv1(out, BiSNN))
        out2 = self.bn2(self.binary_conv2(out1, BiSNN))
        out3 = self.bn3(self.binary_conv3(out2, BiSNN))
        out_sum = out1 + out2 + out3  # multiple shortcuts fusion

        # apply attention if present
        if self.attention is not None:
            out_sum = self.attention(out_sum)

        # integrate & fire
        mem, spk = self.lif(out_sum, mem, spike)

        # apply spike normalization/homeostasis
        if self.use_spikenorm and self.spike_norm is not None:
            spk = self.spike_norm(spk)

        # apply downsample to residual if necessary to match shapes
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.nonlinear(out)

        return out, mem, spk
