// 2013 (c) Mika Pi

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>

#include "LlamaComponent.generated.h"

namespace Internal
{
  class Llama;
}

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class UELLAMA_API ULlamaComponent : public UActorComponent
{
  GENERATED_BODY()
public:
  ULlamaComponent(const FObjectInitializer &ObjectInitializer);
  ~ULlamaComponent();

private:
  std::unique_ptr<Internal::Llama> m_llama;
};
