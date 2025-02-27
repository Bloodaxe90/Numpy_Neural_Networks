import numpy as np


class MaxPool:

    def __init__(self,
                 kernel_size: int = 2,
                 padding: int = 0,
                 stride: int = 2
                 ):

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.padded_input_a: np.ndarray = None
        self.max_indexes = None

    def forward_propagation(self, input_a: np.ndarray) -> np.ndarray:
        batch_size, input_channels, height, width = input_a.shape

        assert self.padding > -1 and self.stride > -1 and self.kernel_size > 0, "Padding, stride or kernal size is invalid"

        output_height: int = int(
            (height + (2 * self.padding) - self.kernel_size) / self.stride) + 1
        output_width: int = int(
            (width + (2 * self.padding) - self.kernel_size) / self.stride) + 1

        output = np.zeros((
            batch_size,
            input_channels,
            output_height,
            output_width,
        ))

        self.max_indexes = np.zeros((
            batch_size,
            input_channels,
            output_height,
            output_width,
            2
        ), dtype=int)

        self.padded_input_a = np.pad(input_a,
                                pad_width=(
                                (0, 0), (0, 0), (self.padding, self.padding),
                                (self.padding, self.padding)),
                                mode="constant",
                                constant_values=0,
                                )

        for batch in range(batch_size):
            for channel in range(input_channels):
                for row in range(output_height):
                    for column in range(output_width):
                        row_start = row * self.stride
                        column_start = column * self.stride
                        local_receptive_feild = self.padded_input_a[
                                    batch,
                                    channel,
                                    row_start: row_start + self.kernel_size,
                                    column_start: column_start + self.kernel_size
                                ]
                        neuron_a = np.max(local_receptive_feild)
                        max_index = np.argwhere(local_receptive_feild == neuron_a)
                        if len(max_index) > 1:
                            max_index = max_index[np.random.choice(len(max_index))]
                        self.max_indexes[batch, channel, row, column] = max_index
                        output[batch, channel, row, column] = neuron_a

        return output

    def backpropagation(self, dldz):
        batch_size, input_channels = self.padded_input_a.shape[0:2]
        output_height, output_width = dldz.shape[-2:]

        padded_masked_dldz = np.zeros_like(
            self.padded_input_a
        )

        for batch in range(batch_size):
            for channel in range(input_channels):
                for row in range(output_height):
                    for column in range(output_width):
                        mask_row, mask_column = self.max_indexes[batch, channel, row, column]
                        padded_masked_dldz[
                            batch, channel, (row * self.stride) + mask_row, (column * self.stride) + mask_column
                        ] += dldz[batch, channel, row, column]


        if self.padding > 0:
            masked_dldz = padded_masked_dldz[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            masked_dldz = padded_masked_dldz

        return masked_dldz