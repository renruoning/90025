// Scratch differential test: verifies parallel_task2 produces byte-identical
// Results to the sequential task2 reference, including a large synthetic
// corpus designed to exercise the parallel (word-bucketed) code path.
// Not part of the graded suite -- delete before submitting if not wanted.

#include "bpe.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <vector>

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
    bpe::parallel_task2(splits, par);
    EXPECT_EQ(tokens_of(seq), tokens_of(par));
}

}  // namespace

TEST(ParallelTask2Diff, EmptyCorpus) { expect_same({}); }

TEST(ParallelTask2Diff, HuggingFaceExample) {
    expect_same({cs("hug", 10), cs("pug", 5), cs("pun", 12), cs("bun", 4),
                 cs("hugs", 5)});
}

TEST(ParallelTask2Diff, OverlappingOccurrences) {
    expect_same({cs("aaaa", 1), cs("aa", 1)});
}

TEST(ParallelTask2Diff, TieBreak) {
    expect_same({cs("ab", 3), cs("abx", 2)});
}

TEST(ParallelTask2Diff, MultipleRounds) {
    expect_same({cs("abc", 1), cs("abcd", 1)});
}

TEST(ParallelTask2Diff, LargeDistinctWordsForcesParallelRound) {
    std::vector<bpe::CharSplit> splits;
    splits.reserve(5000);
    for (int i = 0; i < 5000; ++i) {
        splits.push_back(cs("ab" + std::to_string(i), 1));
    }
    expect_same(splits);
}

TEST(ParallelTask2Diff, ManyWordsWithInternalOverlap) {
    std::vector<bpe::CharSplit> splits;
    splits.reserve(3000);
    for (int i = 0; i < 3000; ++i) {
        splits.push_back(cs("aaaa" + std::to_string(i % 7), 3));
    }
    expect_same(splits);
}
