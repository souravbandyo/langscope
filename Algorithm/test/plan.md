# LangScope Test Plan

## Overview

This document outlines the comprehensive testing strategy for the LangScope Multi-domain LLM Evaluation Framework. The test suite covers unit tests, integration tests, API tests, and performance tests across all modules.

**Framework**: pytest + pytest-asyncio  
**Coverage Target**: 85%+ line coverage  
**Current Coverage**: 47% (494 tests passing)  
**Test Naming Convention**: `test_<module>_<function>_<scenario>`

## Current Test Status

| Module | Tests | Status | Coverage |
|--------|-------|--------|----------|
| `api/` | 92 | ✅ Complete | 45% |
| `database/` | 81 | ✅ Complete | 16% |
| `ground_truth/` | 69 | ✅ Complete | 30% |
| `core/` | 46 | ✅ Complete | 75% |
| `transfer/` | 42 | ✅ Complete | 60% |
| `feedback/` | 41 | ✅ Complete | 35% |
| `federation/` | 38 | ✅ Complete | 25% |
| `evaluation/` | 28 | ✅ Complete | 40% |
| `ranking/` | 26 | ✅ Complete | 35% |
| `domain/` | 12 | ✅ Complete | 50% |
| `config/` | 7 | ✅ Complete | 65% |
| `integration/` | 12 | ✅ Complete | N/A |
| **Total** | **494** | **✅ All Passing** | **47%** |

---

## 1. Core Module Tests (`langscope/core/`)

### 1.1 `rating.py` - TrueSkillRating & DualTrueSkill

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CORE-001 | Create TrueSkillRating with default values | μ=25.0, σ=8.333 |
| CORE-002 | Create TrueSkillRating with custom values | Values match input |
| CORE-003 | Calculate confidence interval (95%) | Returns (μ-1.96σ, μ+1.96σ) |
| CORE-004 | Calculate conservative estimate (μ-3σ) | Correct penalized value |
| CORE-005 | Convert to/from dictionary | Round-trip preserves data |
| CORE-006 | DualTrueSkill initialization | Both raw and cost_adjusted ratings valid |
| CORE-007 | DualTrueSkill update both ratings | Both ratings updated independently |
| CORE-008 | Edge case: σ = 0 | Precision returns inf, no crash |
| CORE-009 | Edge case: Negative μ values | Handled correctly |

### 1.2 `model.py` - LLMModel

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CORE-010 | Create model with required fields | Model created successfully |
| CORE-011 | Create model with all optional fields | All fields populated |
| CORE-012 | Model serialization to dict | Complete dict output |
| CORE-013 | Model deserialization from dict | Reconstructed model matches |
| CORE-014 | Model with cost information | Cost per 1M tokens calculated |
| CORE-015 | Model domain ratings initialization | Empty domain ratings dict |
| CORE-016 | Add domain rating to model | Rating stored correctly |
| CORE-017 | Get non-existent domain rating | Returns None or default |

### 1.3 `dimensions.py` - Multi-dimensional Scoring

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CORE-020 | List all 10 scoring dimensions | All dimensions present |
| CORE-021 | Dimension weight validation | Weights sum to 1.0 |
| CORE-022 | Custom weight configuration | Accepts valid custom weights |
| CORE-023 | Invalid weights (sum ≠ 1) | Raises validation error |
| CORE-024 | Calculate combined score | Weighted average correct |

### 1.4 `constants.py` - Configuration Constants

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CORE-030 | TrueSkill constants defined | All required constants present |
| CORE-031 | Constants have correct types | Type validation passes |

---

## 2. Ranking Module Tests (`langscope/ranking/`)

### 2.1 `trueskill.py` - TrueSkill Algorithm

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| RANK-001 | Two-player match update | Winner μ increases, loser μ decreases |
| RANK-002 | Multi-player match update (5 players) | All ratings updated by rank |
| RANK-003 | Uncertainty reduction after match | All σ values decrease |
| RANK-004 | Tie handling between 2 players | Minimal rating change |
| RANK-005 | Upset victory (low beats high) | Larger rating transfer |
| RANK-006 | Expected outcome (high beats low) | Smaller rating transfer |
| RANK-007 | Draw probability calculation | Returns valid probability [0,1] |
| RANK-008 | Quality function calculation | Returns valid quality score |
| RANK-009 | Conservative estimate ranking | Sorted by μ-3σ |
| RANK-010 | Factor graph update convergence | Updates converge in max iterations |
| RANK-011 | MultiPlayerTrueSkillUpdater initialization | Correct params loaded |
| RANK-012 | Batch update multiple matches | All ratings updated correctly |
| RANK-013 | Edge case: All players tied | No rating changes |
| RANK-014 | Edge case: Single player match | Handled gracefully |

### 2.2 `plackett_luce.py` - Plackett-Luce Ranking

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| RANK-020 | MLE parameter estimation | Converges to valid λ values |
| RANK-021 | Ranking probability calculation | P(A>B>C) computed correctly |
| RANK-022 | Iterative λ update | Monotonic improvement in likelihood |
| RANK-023 | Convert rankings to pairwise | Correct number of pairs generated |
| RANK-024 | Aggregate multiple rankings | Combined λ estimates valid |
| RANK-025 | Partial ranking handling | Incomplete rankings processed |
| RANK-026 | Edge case: Single model ranking | Returns trivial result |
| RANK-027 | Edge case: 100+ model ranking | Scales without overflow |

### 2.3 `cost_adjustment.py` - Cost-Adjusted Rankings

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| RANK-030 | Calculate cost-adjusted score | μ_raw / log(1 + cost) |
| RANK-031 | Free model (cost=0) | Full raw score |
| RANK-032 | Expensive model penalty | Score reduced appropriately |
| RANK-033 | Cost normalization | Consistent across price ranges |
| RANK-034 | Rank by cost-adjusted score | Correct ordering |
| RANK-035 | Edge case: Negative cost | Raises ValueError |
| RANK-036 | Edge case: Very high cost (>$1000) | No overflow, valid result |

### 2.4 `dimension_ranker.py` - Multi-dimensional Ranking

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| RANK-040 | Rank by raw quality | Correct ordering |
| RANK-041 | Rank by latency | Correct ordering (lower better) |
| RANK-042 | Rank by TTFT | Correct ordering |
| RANK-043 | Rank by consistency | Correct ordering |
| RANK-044 | Rank by token efficiency | Correct ordering |
| RANK-045 | Rank by instruction following | Correct ordering |
| RANK-046 | Rank by hallucination resistance | Correct ordering |
| RANK-047 | Rank by long context | Correct ordering |
| RANK-048 | Rank by combined score | Weighted ranking correct |
| RANK-049 | Switch between dimensions | Rankings update correctly |

---

## 3. Evaluation Module Tests (`langscope/evaluation/`)

### 3.1 `match.py` - Match Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| EVAL-001 | Create new match | Match ID generated, valid state |
| EVAL-002 | Add competitors to match | 5-6 competitors added |
| EVAL-003 | Add judges to match | 3-5 judges added |
| EVAL-004 | Record match response | Response stored with metrics |
| EVAL-005 | Record match rankings | Raw and cost rankings stored |
| EVAL-006 | Calculate match outcome | Winners determined correctly |
| EVAL-007 | Match serialization | Full match data preserved |
| EVAL-008 | Match timestamp validation | ISO format, UTC timezone |
| EVAL-009 | Match info bits calculation | Correct information content |
| EVAL-010 | Edge case: Empty match | Graceful handling |
| EVAL-011 | Edge case: Duplicate model IDs | Raises validation error |

### 3.2 `aggregation.py` - Judge Aggregation

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| EVAL-020 | Aggregate unanimous judge votes | Clear winner |
| EVAL-021 | Aggregate split votes (3-2) | Majority wins |
| EVAL-022 | Aggregate with tie | Tie handled correctly |
| EVAL-023 | Weighted judge aggregation | Higher-rated judges weighted more |
| EVAL-024 | Judge weight normalization | Weights sum to 1.0 |
| EVAL-025 | Soft update calculation | Partial win credit assigned |
| EVAL-026 | Detect judge disagreement | Disagreement flagged |
| EVAL-027 | Calculate judge consensus score | 0-1 consensus metric |

### 3.3 `metrics.py` - Performance Metrics

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| EVAL-030 | Calculate latency metric | Correct ms value |
| EVAL-031 | Calculate TTFT metric | Correct ms value |
| EVAL-032 | Calculate token efficiency | Quality per token ratio |
| EVAL-033 | Calculate consistency score | 1/(1+σ) formula |
| EVAL-034 | Calculate instruction following rate | % compliance |
| EVAL-035 | Calculate hallucination resistance | 1 - hallucination_rate |
| EVAL-036 | Calculate long context degradation | quality@max / quality@4K |
| EVAL-037 | Aggregate consistency from N runs | Correct variance calculation |
| EVAL-038 | Edge case: Zero tokens | Handle division by zero |

### 3.4 `penalties.py` - Penalty Calculations

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| EVAL-040 | Apply cost penalty | Correct reduction |
| EVAL-041 | Apply latency penalty | Correct reduction |
| EVAL-042 | Apply inconsistency penalty | Correct reduction |
| EVAL-043 | Apply combined penalties | Multiplicative application |
| EVAL-044 | No penalty for ideal metrics | Full score retained |

---

## 4. Transfer Module Tests (`langscope/transfer/`)

### 4.1 `transfer_learning.py` - Cross-domain Transfer

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| XFER-001 | Calculate domain similarity | Returns [0,1] similarity |
| XFER-002 | Transfer ratings from related domain | Initial μ estimated |
| XFER-003 | High similarity transfer (>0.8) | Lower uncertainty boost |
| XFER-004 | Medium similarity transfer (0.5-0.8) | Moderate uncertainty boost |
| XFER-005 | Low similarity transfer (<0.5) | High uncertainty boost |
| XFER-006 | Multi-domain transfer weighting | Weighted average of sources |
| XFER-007 | Cold start domain initialization | Valid initial ratings |
| XFER-008 | Transfer decay over time | Old data weighted less |
| XFER-009 | No transfer from unrelated domains | Falls back to default |
| XFER-010 | Language-based similarity | Linguistic distance computed |
| XFER-011 | Task-based similarity | Semantic embedding distance |

### 4.2 `correlation.py` - Domain Correlation

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| XFER-020 | Calculate Pearson correlation | Valid r value [-1,1] |
| XFER-021 | Calculate rank correlation | Spearman ρ computed |
| XFER-022 | Build correlation matrix | NxN matrix for N domains |
| XFER-023 | Identify highly correlated domains | Threshold filtering works |
| XFER-024 | Update correlation with new data | Incremental update |
| XFER-025 | Edge case: Insufficient data | Returns uncertainty flag |

### 4.3 `specialist.py` - Specialist Detection

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| XFER-030 | Detect domain specialist | Model identified |
| XFER-031 | Calculate specialist score | High in target, low elsewhere |
| XFER-032 | Rank specialists by strength | Correct ordering |
| XFER-033 | Generalist detection | High across all domains |
| XFER-034 | Specialist threshold tuning | Configurable detection |
| XFER-035 | Multi-domain specialist | Handles overlapping domains |

---

## 5. Feedback Module Tests (`langscope/feedback/`)

### 5.1 `user_feedback.py` - User Feedback Integration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-001 | Create user session | Session ID generated |
| FEED-002 | Record user feedback (A wins) | Feedback stored |
| FEED-003 | Record user feedback (tie) | Tie handled |
| FEED-004 | Session feedback aggregation | All feedback combined |
| FEED-005 | Prediction state tracking | Before/after predictions |
| FEED-006 | Feedback delta calculation | Change tracked |
| FEED-007 | Session expiration | Old sessions cleaned up |
| FEED-008 | Feedback confidence weighting | User reliability factored |

### 5.2 `workflow.py` - Feedback Workflow

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-010 | Start arena session | Session initialized |
| FEED-011 | Generate battle prompts | Domain-appropriate prompts |
| FEED-012 | Process battle result | Ratings updated |
| FEED-013 | Complete arena session | Final rankings generated |
| FEED-014 | Session progress tracking | Battle count updated |
| FEED-015 | Early session termination | Partial results saved |
| FEED-016 | Session resume | Continues from last battle |

### 5.3 `judge_calibration.py` - Judge Calibration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-020 | Calculate judge accuracy | Accuracy metric computed |
| FEED-021 | Detect judge bias | Provider bias flagged |
| FEED-022 | Update judge weights | Accurate judges weighted higher |
| FEED-023 | Judge consensus tracking | Agreement rate tracked |
| FEED-024 | Exclude biased judges | Below-threshold judges removed |
| FEED-025 | Judge recalibration | Weights adjusted over time |

### 5.4 `accuracy.py` - Prediction Accuracy

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-030 | Calculate prediction accuracy | Correct % computed |
| FEED-031 | Track accuracy over time | Rolling average |
| FEED-032 | Domain-specific accuracy | Per-domain tracking |
| FEED-033 | Accuracy confidence interval | CI computed |

### 5.5 `delta.py` - Feedback Delta Tracking

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-040 | Calculate rating delta | Before/after difference |
| FEED-041 | Track delta over session | Cumulative tracking |
| FEED-042 | Significant delta detection | Large changes flagged |
| FEED-043 | Delta normalization | Comparable across domains |

### 5.6 `use_case.py` - Use Case Recommendations

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-050 | Parse use case description | Keywords extracted |
| FEED-051 | Match use case to domains | Relevant domains identified |
| FEED-052 | Generate recommendations | Top N models returned |
| FEED-053 | Custom weight recommendations | User weights applied |
| FEED-054 | Budget-constrained recommendations | Cost filter applied |

### 5.7 `weights.py` - Weight Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FEED-060 | Get default weights | All 10 dimensions covered |
| FEED-061 | Set custom weights | Weights validated and stored |
| FEED-062 | Weight normalization | Sum equals 1.0 |
| FEED-063 | Invalid weight rejection | Negative weights rejected |

---

## 6. Federation Module Tests (`langscope/federation/`)

### 6.1 `judge.py` - Judge Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FED-001 | Select judge pool | 3-5 judges selected |
| FED-002 | Judge diversity enforcement | No same-provider judges |
| FED-003 | Judge qualification check | Only top-quartile eligible |
| FED-004 | Judge rotation | Different judges each match |
| FED-005 | Judge recusal (conflict of interest) | Participant providers excluded |

### 6.2 `selection.py` - Model Selection

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FED-010 | Select match competitors | 5-6 models selected |
| FED-011 | Select case creator | Top-quartile model |
| FED-012 | Select question creator | Top-quartile model |
| FED-013 | Diversity in selection | Multiple providers |
| FED-014 | Stratified selection | Mix of rating levels |

### 6.3 `strata.py` - Stratified Sampling

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FED-020 | Divide models into strata | N strata created |
| FED-021 | Sample from each stratum | Representative sample |
| FED-022 | Rebalance strata over time | Strata boundaries updated |
| FED-023 | Edge case: Too few models | Reduced strata count |

### 6.4 `workflow.py` - Federation Workflow

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FED-030 | Execute full match workflow | Match completed end-to-end |
| FED-031 | Handle LLM API failure | Graceful retry/fallback |
| FED-032 | Timeout handling | Match marked incomplete |
| FED-033 | Concurrent match execution | No race conditions |

### 6.5 `content.py` - Content Generation

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FED-040 | Generate domain-specific case | Case in correct domain |
| FED-041 | Generate evaluation question | Valid question format |
| FED-042 | Content quality validation | Low-quality rejected |
| FED-043 | Content safety filtering | Harmful content blocked |

---

## 7. Domain Module Tests (`langscope/domain/`)

### 7.1 `domain_config.py` - Domain Configuration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DOM-001 | Load domain configuration | Valid config loaded |
| DOM-002 | Create new domain | Domain added to system |
| DOM-003 | Update domain config | Changes persisted |
| DOM-004 | Delete domain | Domain removed |
| DOM-005 | Validate domain schema | Invalid configs rejected |
| DOM-006 | Get domain evaluation criteria | Criteria returned |
| DOM-007 | Get domain transfer paths | Related domains listed |
| DOM-008 | Domain hierarchy traversal | Parent/child relationships |

### 7.2 `domain_manager.py` - Domain Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DOM-010 | List all domains | Complete domain list |
| DOM-011 | Get domain by ID | Correct domain returned |
| DOM-012 | Search domains by keyword | Matching domains found |
| DOM-013 | Get domain leaderboard | Ranked models returned |
| DOM-014 | Domain statistics | Battle count, model count |
| DOM-015 | Domain health check | All metrics computed |

### 7.3 `prompts.py` - Domain Prompts

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DOM-020 | Get judge prompt for domain | Domain-specific prompt |
| DOM-021 | Get case creator prompt | Valid case template |
| DOM-022 | Get question creator prompt | Valid question template |
| DOM-023 | Prompt variable substitution | Variables replaced |

---

## 8. Config Module Tests (`langscope/config/`)

### 8.1 `settings.py` - Application Settings

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CFG-001 | Load settings from env | Environment variables read |
| CFG-002 | Default settings fallback | Defaults used when env missing |
| CFG-003 | Settings validation | Invalid settings rejected |
| CFG-004 | Settings immutability | Cannot modify after load |

### 8.2 `params/` - Parameter Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CFG-010 | Load TrueSkill params | Valid params loaded |
| CFG-011 | Override params at runtime | Override applied |
| CFG-012 | Param caching | Cached values returned |
| CFG-013 | Param validation | Invalid params rejected |
| CFG-014 | Reset to defaults | Defaults restored |

### 8.3 `api_keys.py` - API Key Management

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CFG-020 | Validate API key | Valid key accepted |
| CFG-021 | Reject invalid API key | 401 returned |
| CFG-022 | Rate limit per API key | Limits enforced |
| CFG-023 | API key rotation | Old keys invalidated |

---

## 9. Database Module Tests (`langscope/database/`)

### 9.1 `mongodb.py` - MongoDB Operations

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DB-001 | Connect to MongoDB | Connection established |
| DB-002 | Insert match document | Document inserted |
| DB-003 | Query matches by domain | Filtered results returned |
| DB-004 | Update model ratings | Ratings updated |
| DB-005 | Aggregate leaderboard | Sorted results returned |
| DB-006 | Transaction handling | ACID properties maintained |
| DB-007 | Index performance | Queries use indexes |
| DB-008 | Connection pooling | Pool size respected |
| DB-009 | Reconnection on failure | Auto-reconnect works |
| DB-010 | Bulk operations | Batch insert/update works |

### 9.2 `migrations.py` - Database Migrations

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DB-020 | Run pending migrations | All migrations applied |
| DB-021 | Rollback migration | Previous state restored |
| DB-022 | Migration idempotency | Re-run is safe |
| DB-023 | Schema version tracking | Current version tracked |

### 9.3 `schemas.py` - Database Schemas

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| DB-030 | Validate match schema | Valid match accepted |
| DB-031 | Validate model schema | Valid model accepted |
| DB-032 | Reject invalid schema | Validation errors raised |
| DB-033 | Schema evolution | Backward compatibility |

---

## 10. API Module Tests (`langscope/api/`)

### 10.1 `main.py` - Application Lifecycle

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-001 | Application startup | All routes registered |
| API-002 | Health check endpoint | Returns healthy status |
| API-003 | Root endpoint | API info returned |
| API-004 | Application shutdown | Cleanup executed |
| API-005 | CORS headers | Headers present |
| API-006 | OpenAPI spec generation | Valid spec returned |

### 10.2 `routes/models.py` - Model Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-010 | GET /models | List all models |
| API-011 | GET /models/{id} | Get specific model |
| API-012 | POST /models | Create new model |
| API-013 | PUT /models/{id} | Update model |
| API-014 | DELETE /models/{id} | Delete model |
| API-015 | GET /models with filters | Filtered results |
| API-016 | Model not found | 404 returned |
| API-017 | Invalid model data | 422 validation error |

### 10.3 `routes/domains.py` - Domain Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-020 | GET /domains | List all domains |
| API-021 | GET /domains/{id} | Get specific domain |
| API-022 | POST /domains | Create new domain |
| API-023 | PUT /domains/{id} | Update domain |
| API-024 | DELETE /domains/{id} | Delete domain |
| API-025 | GET /domains/{id}/models | Models in domain |

### 10.4 `routes/matches.py` - Match Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-030 | POST /matches | Create new match |
| API-031 | GET /matches/{id} | Get match details |
| API-032 | GET /matches | List matches with pagination |
| API-033 | POST /matches/{id}/results | Submit match results |
| API-034 | GET /matches/domain/{domain} | Matches by domain |

### 10.5 `routes/leaderboard.py` - Leaderboard Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-040 | GET /leaderboard/{domain} | Domain leaderboard |
| API-041 | GET /leaderboard/{domain}?type=raw | Raw rankings |
| API-042 | GET /leaderboard/{domain}?type=cost | Cost-adjusted rankings |
| API-043 | GET /leaderboard/{domain}?dimension=latency | Latency rankings |
| API-044 | Leaderboard pagination | Paginated results |
| API-045 | Empty domain leaderboard | Empty list returned |

### 10.6 `routes/transfer.py` - Transfer Learning Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-050 | POST /transfer/predict | Transfer prediction |
| API-051 | GET /transfer/similarity | Domain similarity |
| API-052 | GET /transfer/related/{domain} | Related domains |

### 10.7 `routes/specialists.py` - Specialist Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-060 | GET /specialists/{domain} | Domain specialists |
| API-061 | GET /specialists/model/{id} | Model specializations |

### 10.8 `routes/arena.py` - Arena Mode Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-070 | POST /arena/sessions | Start arena session |
| API-071 | GET /arena/sessions/{id} | Get session status |
| API-072 | POST /arena/sessions/{id}/battles | Submit battle result |
| API-073 | POST /arena/sessions/{id}/complete | Complete session |
| API-074 | GET /arena/sessions/{id}/results | Get final results |

### 10.9 `routes/recommendations.py` - Recommendation Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-080 | POST /recommendations | Get recommendations |
| API-081 | Recommendations with budget | Budget-filtered results |
| API-082 | Recommendations with weights | Custom weighted results |

### 10.10 `routes/params.py` - Parameter Endpoints

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-090 | GET /params | Get current params |
| API-091 | PUT /params | Update params |
| API-092 | POST /params/reset | Reset to defaults |

### 10.11 `middleware.py` - Middleware Tests

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-100 | AuthMiddleware - valid key | Request proceeds |
| API-101 | AuthMiddleware - invalid key | 401 returned |
| API-102 | AuthMiddleware - missing key | 401 returned |
| API-103 | RateLimitMiddleware - under limit | Request proceeds |
| API-104 | RateLimitMiddleware - over limit | 429 returned |
| API-105 | RequestLoggingMiddleware | Request logged |
| API-106 | Middleware chain order | Correct execution order |

### 10.12 `dependencies.py` - Dependency Injection

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-110 | get_db dependency | Database instance returned |
| API-111 | Cleanup on shutdown | Resources released |

---

## 11. Integration Tests

### 11.1 End-to-End Match Flow

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| INT-001 | Full match execution | Match created → executed → results stored → ratings updated |
| INT-002 | Match with 5 competitors | All 5 models compete and rated |
| INT-003 | Match with 3 judges | Aggregated judgement applied |
| INT-004 | Match affects leaderboard | Leaderboard reflects new results |

### 11.2 Arena Session Flow

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| INT-010 | Complete arena session | User runs 10 battles, gets results |
| INT-011 | Arena affects predictions | User feedback improves predictions |
| INT-012 | Arena rating updates | Ratings updated after session |

### 11.3 Transfer Learning Flow

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| INT-020 | Cold start new domain | Transfer from related domains |
| INT-021 | Specialist detection after matches | Specialists identified |
| INT-022 | Cross-domain correlation update | Correlation matrix updated |

### 11.4 API Integration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| INT-030 | Full API CRUD cycle | Create → Read → Update → Delete |
| INT-031 | API rate limiting | Limits enforced over time |
| INT-032 | Concurrent API requests | No race conditions |

---

## 12. Performance Tests

### 12.1 Load Testing

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| PERF-001 | 100 concurrent API requests | <500ms p95 latency |
| PERF-002 | 1000 match insertions | <5s total time |
| PERF-003 | Leaderboard query with 1000 models | <200ms response |
| PERF-004 | TrueSkill update for 100 matches | <1s total time |

### 12.2 Scalability Testing

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| PERF-010 | 10,000 models in database | Queries remain fast |
| PERF-011 | 100,000 matches in database | Aggregations complete |
| PERF-012 | 50 concurrent arena sessions | System stable |

### 12.3 Memory Testing

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| PERF-020 | Memory usage under load | <1GB steady state |
| PERF-021 | No memory leaks | Memory stable over time |

---

## 13. Security Tests

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| SEC-001 | SQL injection prevention | Attacks blocked |
| SEC-002 | NoSQL injection prevention | Attacks blocked |
| SEC-003 | XSS prevention | Scripts escaped |
| SEC-004 | Authentication bypass | All protected routes secured |
| SEC-005 | Rate limiting bypass | Limits enforced |
| SEC-006 | Sensitive data exposure | API keys not logged |

---

## 14. Test Execution Plan

### Phase 1: Unit Tests (Days 1-3) ✅ COMPLETED
- ✅ Core module tests (46 tests)
- ✅ Ranking module tests (26 tests)
- ✅ Evaluation module tests (28 tests)

### Phase 2: Unit Tests Continued (Days 4-6) ✅ COMPLETED
- ✅ Transfer module tests (42 tests)
- ✅ Feedback module tests (41 tests)
- ✅ Federation module tests (38 tests)
- ✅ Domain module tests (12 tests)
- ✅ Config module tests (7 tests)
- ✅ Database module tests (81 tests)

### Phase 3: API Tests (Days 7-9) ✅ COMPLETED
- ✅ All API route tests (92 tests)
- ✅ Middleware tests
- ✅ Schema validation tests
- ✅ Error handling tests

### Phase 4: Integration Tests (Days 10-12) ✅ COMPLETED
- ✅ End-to-end flows (12 tests)
- ✅ Cross-module interactions

### Phase 5: Performance & Security (Days 13-14) ⏳ PENDING
- ⏳ Load testing
- ⏳ Security testing
- ⏳ Final regression

---

## 15. Test Infrastructure

### Required Dependencies
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
httpx>=0.23.0  # For async API testing
mongomock>=4.0.0  # Mock MongoDB
factory-boy>=3.2.0  # Test data factories
faker>=18.0.0  # Fake data generation
locust>=2.15.0  # Load testing
```

### Test Configuration (`pytest.ini`)
```ini
[pytest]
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = --cov=langscope --cov-report=html --cov-fail-under=85
markers =
    unit: Unit tests
    integration: Integration tests
    api: API tests
    slow: Slow tests (>1s)
    performance: Performance tests
```

### Test Directory Structure (Current)
```
test/
├── conftest.py              # Shared fixtures (mock_db, sample models, etc.)
├── plan.md                  # This test plan document
├── test_api.py              # API endpoint tests (92 tests)
│   ├── TestRootAndHealth    # Health check endpoints
│   ├── TestModelsAPI        # Model CRUD operations
│   ├── TestLeaderboardAPI   # Ranking endpoints (10 dimensions)
│   ├── TestTransferAPI      # Cross-domain transfer endpoints
│   ├── TestSpecialistsAPI   # Specialist detection endpoints
│   ├── TestArenaAPI         # User feedback session endpoints
│   ├── TestGroundTruthAPI   # Ground truth evaluation endpoints
│   ├── TestMatchesAPI       # Match execution endpoints
│   ├── TestMonitoringAPI    # System monitoring endpoints
│   ├── TestRecommendationsAPI  # Use-case recommendations
│   ├── TestParamsAPI        # Parameter management endpoints
│   ├── TestBaseModelsAPI    # Base model management endpoints
│   ├── TestDeploymentsAPI   # Deployment management endpoints
│   ├── TestSelfHostedAPI    # Self-hosted deployment endpoints
│   ├── TestBenchmarksAPI    # Benchmark endpoints
│   ├── TestMiddleware       # Auth and rate limiting tests
│   └── TestAPIErrorHandling # Error handling tests
├── test_core.py             # Core module tests (46 tests)
│   ├── TestTrueSkillRating  # Rating calculations
│   ├── TestMultiDimensionalTrueSkill  # 10D rating system
│   ├── TestLLMModel         # Model data structures
│   ├── TestDimensions       # Dimension enums and weights
│   └── TestConstants        # Default values
├── test_database.py         # Database operation tests (81 tests)
│   ├── TestConnection       # MongoDB connection handling
│   ├── TestModelCRUD        # Model operations
│   ├── TestMatchCRUD        # Match operations
│   ├── TestDomainCRUD       # Domain operations
│   ├── TestParameterCRUD    # Parameter operations
│   └── TestTimeSeries       # Rating/price history
├── test_transfer.py         # Transfer learning tests (42 tests)
│   ├── TestCorrelationLearner      # Bayesian correlation
│   ├── TestTransferLearning        # Rating transfer
│   └── TestSpecialistDetector      # Specialist/generalist detection
├── test_feedback.py         # User feedback tests (41 tests)
│   ├── TestUserSession      # Session management
│   ├── TestFeedbackDelta    # Delta computation
│   ├── TestAccuracy         # Prediction accuracy metrics
│   ├── TestUseCaseProfile   # Use-case adjustments
│   └── TestWorkflow         # Full feedback workflow
├── test_federation.py       # Federation module tests (38 tests)
│   ├── TestJudgeRankingValidator   # Ranking validation
│   ├── TestStrata           # Stratum assignment
│   ├── TestSelection        # Swiss pairing, judge selection
│   └── TestContentGenerator # Content generation
├── test_ground_truth.py     # Ground truth tests (69 tests)
│   ├── TestMetricRegistry   # Metric definitions
│   ├── TestJudges           # Domain-specific judges
│   ├── TestSampling         # Stratified sampling
│   └── TestWorkflow         # Evaluation workflow
├── test_evaluation.py       # Match evaluation tests (28 tests)
├── test_ranking.py          # Ranking algorithm tests (26 tests)
├── test_domain.py           # Domain management tests (12 tests)
├── test_config.py           # Configuration tests (7 tests)
└── test_integration.py      # Integration tests (12 tests)
```

---

## 16. Fixtures & Mocks

### Core Fixtures (`conftest.py`)

#### Database Fixtures
- `mock_db`: Mocked MongoDB connection with all CRUD operations
- `mock_db_with_data`: Database pre-populated with sample data

#### Model Fixtures
- `sample_model`: Pre-configured LLMModel instance
- `sample_models`: List of 5 test models with varying ratings
- `sample_rating`: TrueSkillRating with known values (μ=25.0, σ=8.333)
- `multi_dimensional_rating`: MultiDimensionalTrueSkill with all 10 dimensions

#### Match Fixtures
- `sample_match`: Complete Match object with participants and results
- `sample_match_result`: MatchResult with rankings and metrics

#### Domain Fixtures
- `sample_domain`: Domain configuration for "coding"
- `sample_domains`: List of test domains

#### API Fixtures
- `client`: FastAPI TestClient with mocked dependencies
- `api_key_headers`: Valid authentication headers (`{"X-API-Key": "test-api-key"}`)

#### Transfer Learning Fixtures
- `correlation_learner`: CorrelationLearner with test data
- `transfer_learner`: TransferLearning instance
- `specialist_detector`: SpecialistDetector instance

#### Feedback Fixtures
- `user_session`: UserSession with test battles
- `use_case_manager`: UseCaseAdjustmentManager instance
- `feedback_workflow`: UserFeedbackWorkflow instance

### Example Fixture Usage

```python
def test_model_creation(sample_model):
    """Use the sample_model fixture."""
    assert sample_model.model_id is not None
    assert sample_model.name == "Test Model"

def test_api_endpoint(client, api_key_headers):
    """Use API fixtures."""
    response = client.get("/models", headers=api_key_headers)
    assert response.status_code == 200

def test_transfer_learning(correlation_learner):
    """Use transfer learning fixtures."""
    correlation = correlation_learner.get_correlation("coding", "math")
    assert 0 <= correlation <= 1
```

---

## Appendix: Test Coverage Requirements

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| langscope/core | 90% | 75% | 🔶 In Progress |
| langscope/ranking | 90% | 35% | 🔶 In Progress |
| langscope/evaluation | 85% | 40% | 🔶 In Progress |
| langscope/transfer | 85% | 60% | 🔶 In Progress |
| langscope/feedback | 85% | 35% | 🔶 In Progress |
| langscope/federation | 80% | 25% | 🔶 In Progress |
| langscope/domain | 85% | 50% | 🔶 In Progress |
| langscope/config | 80% | 65% | 🔶 In Progress |
| langscope/database | 75% | 16% | 🔶 In Progress |
| langscope/api | 85% | 45% | 🔶 In Progress |
| langscope/ground_truth | 80% | 30% | 🔶 In Progress |
| **Overall** | **85%** | **47%** | 🔶 **In Progress** |

### Coverage Improvement Roadmap

To reach 85% coverage, focus on:

1. **Database Module** (16% → 75%): Add tests for actual MongoDB operations
2. **Federation Module** (25% → 80%): Add workflow and content generation tests
3. **Ground Truth Module** (30% → 80%): Add judge and sampling tests
4. **Ranking Module** (35% → 90%): Add Plackett-Luce and dimension ranker tests
5. **Feedback Module** (35% → 85%): Add judge calibration and workflow tests

