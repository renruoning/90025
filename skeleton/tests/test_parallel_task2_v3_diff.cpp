// Differential test for the experimental parallel_task2_v3 (see
// src/parallel_task2_v3.cpp) against the sequential task2 reference.
// Mirrors test_parallel_task2_diff.cpp's coverage. Not part of the graded
// suite; parallel_task2_v3 is not wired into the real pipeline.

#include "bpe.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <vector>

namespace bpe {
void parallel_task2_v3(const std::vector<CharSplit>& splits, Results& results);
}  // namespace bpe

namespace {

bpe::CharSplit cs(const std::string& word, std::size_t count) {
    return {std::vector<bpe::Byte>(word.begin(), word.end()), count};
}

using StrCount = std::pair<std::string, std::size_t>;

std::vector<StrCount> tokens_of(const bpe::Results& r) {
    std::vector<StrCount> out;
    for (const bpe::TokenCount& t : r.tokens) {
        out.emplace_back(std::string(t.token.begin(), t.token.end()), t.count);
    }
    return out;
}

void expect_same(const std::vector<bpe::CharSplit>& splits) {
    bpe::Results seq;
    bpe::task2(splits, seq);
    bpe::Results par;
    bpe::parallel_task2_v3(splits, par);
    EXPECT_EQ(tokens_of(seq), tokens_of(par));
}

}  // namespace

TEST(ParallelTask2V3Diff, EmptyCorpus) { expect_same({}); }

TEST(ParallelTask2V3Diff, HuggingFaceExample) {
    expect_same({cs("hug", 10), cs("pug", 5), cs("pun", 12), cs("bun", 4),
                 cs("hugs", 5)});
}

TEST(ParallelTask2V3Diff, OverlappingOccurrences) {
    expect_same({cs("aaaa", 1), cs("aa", 1)});
}

TEST(ParallelTask2V3Diff, TieBreak) {
    expect_same({cs("ab", 3), cs("abx", 2)});
}

TEST(ParallelTask2V3Diff, MultipleRounds) {
    expect_same({cs("abc", 1), cs("abcd", 1)});
}

TEST(ParallelTask2V3Diff, LargeDistinctWordsForcesParallelRound) {
    std::vector<bpe::CharSplit> splits;
    splits.reserve(25000);
    for (int i = 0; i < 25000; ++i) {
        splits.push_back(cs("ab" + std::to_string(i), 1));
    }
    expect_same(splits);
}

TEST(ParallelTask2V3Diff, ManyWordsWithInternalOverlap) {
    // Each word contributes 3 occurrences of (a,a) to the first round --
    // exercises the "own pair" batching path with words_merged*3 !=
    // positions.size(), i.e. multiple occurrences per word.
    std::vector<bpe::CharSplit> splits;
    splits.reserve(8000);
    for (int i = 0; i < 8000; ++i) {
        splits.push_back(cs("aaaa" + std::to_string(i % 7), 3));
    }
    expect_same(splits);
}

TEST(ParallelTask2V3Diff, HighInternalOverlapDominatesRound) {
    // Fewer distinct words, but each with many repeated characters, so a
    // single word contributes many occurrences per round -- the scenario
    // where the "own pair" batching should matter most.
    std::vector<bpe::CharSplit> splits;
    splits.reserve(30000);
    std::string long_word(50, 'a');
    for (int i = 0; i < 30000; ++i) {
        splits.push_back(cs(long_word + std::to_string(i % 3), 2));
    }
    expect_same(splits);
}
