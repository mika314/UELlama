// 2023 (c) Mika Pi

#include "UELlama/LlamaComponent.h"
#include <atomic>
#include <deque>
#include <thread>

#define GGML_CUDA_DMMV_X 64
#define GGML_CUDA_F16
#define GGML_CUDA_MMV_Y 2
#define GGML_USE_CUBLAS
#define GGML_USE_K_QUANTS
#define K_QUANTS_PER_ITERATION 2
#include "llama.h"

namespace
{
  class Q
  {
  public:
    auto enqueue(std::function<void()>) -> void;
    auto processQ() -> bool;

  private:
    std::deque<std::function<void()>> q;
    std::mutex mutex;
  };

  auto Q::enqueue(std::function<void()> v) -> void
  {
    std::lock_guard<std::mutex> l(mutex);
    q.emplace_back(std::move(v));
  }

  auto Q::processQ() -> bool
  {
    auto v = [this]() -> std::function<void()> {
      std::lock_guard<std::mutex> l(mutex);
      if (q.empty())
        return nullptr;
      auto v = std::move(q.front());
      q.pop_front();
      return v;
    }();
    if (!v)
      return false;
    v();
    return true;
  }
  // TODO: not great allocating this every time
  std::vector<llama_token> my_llama_tokenize(struct llama_context *ctx,
                                             const std::string &text,
                                             bool add_bos)
  {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    const int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    res.resize(n);

    return res;
  }
  const auto n_threads = 8;
} // namespace

namespace Internal
{
  class Llama
  {
  public:
    Llama();
    ~Llama();

    auto activate(bool bReset, FString prompt) -> void;
    auto deactivate() -> void;
    auto insertPrompt(FString v) -> void;
    auto process() -> void;

    std::function<void(FString)> tokenCb;

  private:
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    Q qMainToThread;
    Q qThreadToMain;
    std::atomic_bool running = true;
    std::thread thread;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> embd;
    int n_past = 0;
    std::vector<llama_token> last_n_tokens;
    int n_consumed = 0;
    bool eos = false;

    auto threadRun() -> void;
    auto unsafeActivate(bool bReset, FString prompt) -> void;
    auto unsafeDeactivate() -> void;
    auto unsafeInsertPrompt(FString) -> void;
  };

  auto Llama::insertPrompt(FString v) -> void
  {
    qMainToThread.enqueue([this, v = std::move(v)]() mutable { unsafeInsertPrompt(std::move(v)); });
  }

  auto Llama::unsafeInsertPrompt(FString v) -> void
  {
    auto stdV = std::string(" ") + TCHAR_TO_UTF8(*v);
    auto line_inp = my_llama_tokenize(ctx, stdV.c_str(), false);
    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
  }

  Llama::Llama() : thread([this]() { threadRun(); }) {}

  auto Llama::threadRun() -> void
  {
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread is running"), this);
    const auto n_predict = -1;
    const auto n_keep = 0;
    const auto n_batch = 512;
    while (running)
    {
      while (qMainToThread.processQ())
        ;
      if (!model)
      {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(200ms);
        continue;
      }

      if (eos && embd_inp.empty())
        continue;
      eos = false;

      const auto n_ctx = llama_n_ctx(ctx);
      if (embd.size() > 0)
      {
        // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
        // --prompt or --file which uses the same value.
        auto max_embd_size = n_ctx - 4;
        // Ensure the input doesn't exceed the context size by truncating embd if necessary.
        if ((int)embd.size() > max_embd_size)
        {
          auto skipped_tokens = embd.size() - max_embd_size;
          UE_LOG(LogTemp,
                 Error,
                 TEXT("<<input too long: skipped %zu token%s>>"),
                 skipped_tokens,
                 skipped_tokens != 1 ? "s" : "");
          embd.resize(max_embd_size);
        }

        // infinite text generation via context swapping
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via n_past)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
        if (n_past + (int)embd.size() > n_ctx)
        {
          if (n_predict == -2)
          {
            UE_LOG(LogTemp, Error, TEXT("context full, stopping generation"));
            unsafeDeactivate();
            continue;
          }

          const int n_left = n_past - n_keep;
          // always keep the first token - BOS
          n_past = std::max(1, n_keep);

          // insert n_left/2 tokens at the start of embd from last_n_tokens
          embd.insert(embd.begin(),
                      last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                      last_n_tokens.end() - embd.size());

          // printf("\n---\n");
          // printf("resetting: '");
          // for (int i = 0; i < (int) embd.size(); i++) {
          //     printf("%s", llama_token_to_str(ctx, embd[i]));
          // }
          // printf("'\n");
          // printf("\n---\n");
        }

        // evaluate tokens in batches
        // embd is typically prepared beforehand to fit within a batch, but not always

        for (int i = 0; i < (int)embd.size(); i += n_batch)
        {
          int n_eval = (int)embd.size() - i;
          if (n_eval > n_batch)
          {
            n_eval = n_batch;
          }
          if (llama_eval(ctx, &embd[i], n_eval, n_past, n_threads))
          {
            UE_LOG(LogTemp, Error, TEXT("failed to eva"));
            unsafeDeactivate();
            continue;
          }
          n_past += n_eval;
        }
      }

      embd.clear();

      if ((int)embd_inp.size() <= n_consumed)
      {
        // out of user input, sample next token
        const float temp = 0.80f;
        const int32_t top_k = 40;
        const float top_p = 0.95f;
        const float tfs_z = 1.00f;
        const float typical_p = 1.00f;
        const int32_t repeat_last_n = 64;
        const float repeat_penalty = 1.10f;
        const float alpha_presence = 0.00f;
        const float alpha_frequency = 0.00f;
        const int mirostat = 0;
        const float mirostat_tau = 5.f;
        const float mirostat_eta = 0.1f;
        const bool penalize_nl = true;

        llama_token id = 0;

        {
          auto logits = llama_get_logits(ctx);
          auto n_vocab = llama_n_vocab(ctx);

          std::vector<llama_token_data> candidates;
          candidates.reserve(n_vocab);
          for (llama_token token_id = 0; token_id < n_vocab; token_id++)
          {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
          }

          llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

          // Apply penalties
          float nl_logit = logits[llama_token_nl()];
          auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
          llama_sample_repetition_penalty(ctx,
                                          &candidates_p,
                                          last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                          last_n_repeat,
                                          repeat_penalty);
          llama_sample_frequency_and_presence_penalties(ctx,
                                                        &candidates_p,
                                                        last_n_tokens.data() + last_n_tokens.size() -
                                                          last_n_repeat,
                                                        last_n_repeat,
                                                        alpha_frequency,
                                                        alpha_presence);
          if (!penalize_nl)
          {
            logits[llama_token_nl()] = nl_logit;
          }

          if (temp <= 0)
          {
            // Greedy sampling
            id = llama_sample_token_greedy(ctx, &candidates_p);
          }
          else
          {
            if (mirostat == 1)
            {
              static float mirostat_mu = 2.0f * mirostat_tau;
              const int mirostat_m = 100;
              llama_sample_temperature(ctx, &candidates_p, temp);
              id = llama_sample_token_mirostat(
                ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            }
            else if (mirostat == 2)
            {
              static float mirostat_mu = 2.0f * mirostat_tau;
              llama_sample_temperature(ctx, &candidates_p, temp);
              id = llama_sample_token_mirostat_v2(
                ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            }
            else
            {
              // Temperature sampling
              llama_sample_top_k(ctx, &candidates_p, top_k, 1);
              llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
              llama_sample_typical(ctx, &candidates_p, typical_p, 1);
              llama_sample_top_p(ctx, &candidates_p, top_p, 1);
              llama_sample_temperature(ctx, &candidates_p, temp);
              id = llama_sample_token(ctx, &candidates_p);
            }
          }
          // printf("`%d`", candidates_p.size);

          last_n_tokens.erase(last_n_tokens.begin());
          last_n_tokens.push_back(id);
        }

        // add it to the context
        embd.push_back(id);
      }
      else
      {
        // some user input remains from prompt or interaction, forward it to processing
        while ((int)embd_inp.size() > n_consumed)
        {
          embd.push_back(embd_inp[n_consumed]);
          last_n_tokens.erase(last_n_tokens.begin());
          last_n_tokens.push_back(embd_inp[n_consumed]);
          ++n_consumed;
          if ((int)embd.size() >= n_batch)
          {
            break;
          }
        }
      }

      // display text
      for (auto id : embd)
      {
        FString token = UTF8_TO_TCHAR(llama_token_to_str(ctx, id));
        qThreadToMain.enqueue([token = std::move(token), this]() {
          if (!tokenCb)
            return;
          tokenCb(std::move(token));
        });
      }

      if (!embd.empty() && embd.back() == llama_token_eos())
      {
        eos = true;
      }
    }
    unsafeDeactivate();
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread stopped"), this);
  }

  Llama::~Llama()
  {
    running = false;
    thread.join();
  }

  auto Llama::process() -> void
  {
    while (qThreadToMain.processQ())
      ;
  }

  auto Llama::activate(bool bReset, FString prompt) -> void
  {
    qMainToThread.enqueue([this, bReset, prompt = std::move(prompt)]() mutable {
      unsafeActivate(bReset, std::move(prompt));
    });
  }

  auto Llama::deactivate() -> void
  {
    qMainToThread.enqueue([this]() { unsafeDeactivate(); });
  }

  auto Llama::unsafeActivate(bool bReset, FString prompt) -> void
  {
    UE_LOG(LogTemp, Warning, TEXT("%p Loading LLM model %p bReset: %d"), this, model, bReset);
    if (bReset)
      Llama::unsafeDeactivate();
    if (model)
      return;
    auto lparams = []() {
      auto lparams = llama_context_default_params();
      // -eps 1e-5 -t 8 -ngl 50
      lparams.rms_norm_eps = 1e-5;
      lparams.n_gpu_layers = 50;
      lparams.n_ctx = 4096;
      lparams.seed = time(nullptr);
      return lparams;
    }();
    model =
      llama_load_model_from_file("/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin", lparams);
    if (!model)
    {
      UE_LOG(LogTemp, Error, TEXT("%p unable to load model"), this);
      unsafeDeactivate();
      return;
    }
    ctx = llama_new_context_with_model(model, lparams);

    // tokenize the prompt
    std::string stdPrompt = std::string(" ") + TCHAR_TO_UTF8(*prompt);
    embd_inp = my_llama_tokenize(ctx, stdPrompt.c_str(), true);
    const int n_ctx = llama_n_ctx(ctx);

    if ((int)embd_inp.size() > n_ctx - 4)
    {
      UE_LOG(
        LogTemp, Error, TEXT("prompt is too long (%d tokens, max %d)"), (int)embd_inp.size(), n_ctx - 4);
      unsafeDeactivate();
      return;
    }

    // do one empty run to warm up the model
    {
      const std::vector<llama_token> tmp = {
        llama_token_bos(),
      };
      llama_eval(ctx, tmp.data(), tmp.size(), 0, n_threads);
      llama_reset_timings(ctx);
    }
    last_n_tokens.resize(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_consumed = 0;
  }

  auto Llama::unsafeDeactivate() -> void
  {
    UE_LOG(LogTemp, Warning, TEXT("%p Unloading LLM model %p"), this, model);
    if (!model)
      return;
    llama_print_timings(ctx);
    llama_free(ctx);
    ctx = nullptr;
    llama_free_model(model);
    model = nullptr;
  }
} // namespace Internal

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
  : UActorComponent(ObjectInitializer), llama(std::make_unique<Internal::Llama>())
{
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.bStartWithTickEnabled = true;
  llama->tokenCb = [this](FString NewToken) { OnNewTokenGenerated.Broadcast(std::move(NewToken)); };
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset)
{
  Super::Activate(bReset);
  llama->activate(bReset, prompt);
}

void ULlamaComponent::Deactivate()
{
  llama->deactivate();
  Super::Deactivate();
}

auto ULlamaComponent::TickComponent(float DeltaTime,
                                    enum ELevelTick TickType,
                                    FActorComponentTickFunction *ThisTickFunction) -> void
{
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  llama->process();
}

void ULlamaComponent::InsertPrompt(const FString &v)
{
  llama->insertPrompt(v);
}
