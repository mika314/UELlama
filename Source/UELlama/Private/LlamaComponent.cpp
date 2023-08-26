// 2013 (c) Mika Pi

#include "UELlama/LlamaComponent.h"
#define GGML_CUDA_DMMV_X 64
#define GGML_CUDA_F16
#define GGML_CUDA_MMV_Y 2
#define GGML_USE_CUBLAS
#define GGML_USE_K_QUANTS
#define K_QUANTS_PER_ITERATION 2
#include "llama.h"

namespace Internal
{
  class Llama
  {
  public:
    Llama();
    ~Llama();

  private:
    llama_model *model;
    llama_context *ctx;
    llama_context *ctx_guidance = NULL;
  };

  Llama::Llama()
    : model(
        llama_load_model_from_file("/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin", []() {
          auto lparams = llama_context_default_params();
          return lparams;
        }()))
  {
  }

  Llama::~Llama()
  {
    llama_free_model(model);
  }
} // namespace Internal

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
  : UActorComponent(ObjectInitializer), m_llama(std::make_unique<Internal::Llama>())
{
}

ULlamaComponent::~ULlamaComponent() = default;
