// 2023 (c) Mika Pi

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>

#include "LlamaComponent.generated.h"

namespace Internal
{
  class Llama;
}

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGenerated, FString, NewToken);

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class UELLAMA_API ULlamaComponent : public UActorComponent
{
  GENERATED_BODY()
public:
  ULlamaComponent(const FObjectInitializer &ObjectInitializer);
  ~ULlamaComponent();

  virtual void Activate(bool bReset) override;
  virtual void Deactivate() override;
  virtual void TickComponent(float DeltaTime,
                             ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

  UPROPERTY(BlueprintAssignable)
  FOnNewTokenGenerated OnNewTokenGenerated;

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FString prompt = "Hello";

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FString pathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  TArray<FString> stopSequences;

  UFUNCTION(BlueprintCallable)
  void InsertPrompt(const FString &v);

private:
  std::unique_ptr<Internal::Llama> llama;
};
