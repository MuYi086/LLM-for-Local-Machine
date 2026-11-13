// 无模型验证：主/草稿/检查点容量计算、超限拒绝、前缀去重和淘汰。
#include "server-task.h"
#include <cassert>
#include <iostream>

static server_prompt make_prompt(int start) {
    server_prompt prompt;
    for (int i = 0; i < 100; ++i) {
        prompt.tokens.push_back(start + i);
    }
    auto checkpoint = std::make_shared<common_prompt_checkpoint>();
    checkpoint->data_tgt.resize(65536);
    checkpoint->data_dft.resize(65536);
    prompt.checkpoints.push_back(checkpoint);
    return prompt;
}

int main() {
    server_prompt_cache cache(1, 10000);
    auto prompt = make_prompt(10);
    auto * state = cache.alloc(prompt, 524288, 131072);
    assert(state && cache.states.size() == 1);
    assert(state->data.main.size() == 524288 && state->data.drft.size() == 131072);
    assert(state->size() == 786432);
    assert(cache.alloc(prompt, 524288, 131072) == nullptr);
    auto other = make_prompt(1000);
    assert(cache.alloc(other, 1048576, 1) == nullptr);
    assert(cache.states.size() == 1);
    prompt.tokens.push_back(999);
    assert(cache.alloc(prompt, 524288, 131072));
    assert(cache.states.size() == 1 && cache.states.front().prompt.n_tokens() == 101);
    assert(cache.alloc(other, 524288, 131072));
    assert(cache.states.size() == 1 && cache.states.front().prompt.tokens.get_common_prefix(other.tokens) == 100);
    cache.update();
    assert(cache.size() <= 1048576);
    std::cout << "target/draft/checkpoint budget, prefix replacement and eviction passed\n";
}
