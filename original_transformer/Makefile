CC = g++
CFLAGS = -Wall -Wextra -g -Wno-sign-compare -std=c++11 -I/users/LiuQun/Transformer_Accelerator/original_transformer/accelerator -I/users/LiuQun/Transformer_Accelerator/original_transformer/transformer_layers

# Add the .cc files here
SOURCES = transformer.cpp \
          transformer_layers/transformerBlock.cc \
          transformer_layers/addNorm.cc \
          transformer_layers/dense.cc \
          transformer_layers/selfattention.cc \
          transformer_layers/softmax.cc \
          transformer_layers/transpose.cc \
          accelerator/smm_gem.cpp

transformer: $(SOURCES)
	$(CC) $(CFLAGS) -o transformer $(SOURCES)

.PHONY: clean
clean:
	rm -f transformer
