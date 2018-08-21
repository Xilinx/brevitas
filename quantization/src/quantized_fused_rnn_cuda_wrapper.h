int quantized_fused_lstm_Cudaforward(
    THCudaTensor *input,
    THCudaTensor *hidden,
    THCudaTensor *bias1,
    THCudaTensor *bias2,
    THCudaTensor *cx,
    THCudaTensor *hy,
    THCudaTensor *cy,
    THCudaTensor *quantizationBitWidth);

int quantized_fused_lstm_nobias_Cudaforward(
    THCudaTensor *input,
    THCudaTensor *hidden,
    THCudaTensor *cx,
    THCudaTensor *hy,
    THCudaTensor *cy,
    THCudaTensor *quantizationBitWidth);

int quantized_fused_lstm_Cudabackward(
   THCudaTensor *storage,
   THCudaTensor *gradInGates,
   THCudaTensor *cx,
   THCudaTensor *cy,
   THCudaTensor *gradOutput,
   THCudaTensor *gradOutputCell,
   THCudaTensor *gradInputCx);

int quantized_fused_gru_Cudaforward(
   THCudaTensor *input,
   THCudaTensor *hidden,
   THCudaTensor *bias1,
   THCudaTensor *bias2,
   THCudaTensor *hx,
   THCudaTensor *hy,
   THCudaTensor *storage);

int quantized_fused_gru_nobias_Cudaforward(
   THCudaTensor *input,
   THCudaTensor *hidden,
   THCudaTensor *hx,
   THCudaTensor *hy,
   THCudaTensor *storage);

int quantized_fused_gru_Cudabackward(
   THCudaTensor *gradInInput,
   THCudaTensor *gradInHidden,
   THCudaTensor *gradOutput,
   THCudaTensor *gradInputHx,
   THCudaTensor *storage);

int quantized_fused_lstm_CudaDoubleforward(
    THCudaDoubleTensor *input,
    THCudaDoubleTensor *hidden,
    THCudaDoubleTensor *bias1,
    THCudaDoubleTensor *bias2,
    THCudaDoubleTensor *cx,
    THCudaDoubleTensor *hy,
    THCudaDoubleTensor *cy,
    THCudaDoubleTensor *quantizationBitWidth);

int quantized_fused_lstm_nobias_CudaDoubleforward(
    THCudaDoubleTensor *input,
    THCudaDoubleTensor *hidden,
    THCudaDoubleTensor *cx,
    THCudaDoubleTensor *hy,
    THCudaDoubleTensor *cy,
    THCudaDoubleTensor *quantizationBitWidth);

int quantized_fused_lstm_CudaDoublebackward(
   THCudaDoubleTensor *storage,
   THCudaDoubleTensor *gradInGates,
   THCudaDoubleTensor *cx,
   THCudaDoubleTensor *cy,
   THCudaDoubleTensor *gradOutput,
   THCudaDoubleTensor *gradOutputCell,
   THCudaDoubleTensor *gradInputCx);

int quantized_fused_gru_CudaDoubleforward(
   THCudaDoubleTensor *input,
   THCudaDoubleTensor *hidden,
   THCudaDoubleTensor *bias1,
   THCudaDoubleTensor *bias2,
   THCudaDoubleTensor *hx,
   THCudaDoubleTensor *hy,
   THCudaDoubleTensor *storage);

int quantized_fused_gru_nobias_CudaDoubleforward(
   THCudaDoubleTensor *input,
   THCudaDoubleTensor *hidden,
   THCudaDoubleTensor *hx,
   THCudaDoubleTensor *hy,
   THCudaDoubleTensor *storage);

int quantized_fused_gru_CudaDoublebackward(
   THCudaDoubleTensor *gradInInput,
   THCudaDoubleTensor *gradInHidden,
   THCudaDoubleTensor *gradOutput,
   THCudaDoubleTensor *gradInputHx,
   THCudaDoubleTensor *storage);
