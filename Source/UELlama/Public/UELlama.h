// Copyright (c) 2023 Mika Pi

#pragma once

#include <CoreMinimal.h>
#include <Modules/ModuleManager.h>

class FUELlamaModule final : public IModuleInterface
{
public:
  void StartupModule() final;
  void ShutdownModule() final;

private:
};
