import type { JudgeControlItem, JudgeVerdict } from '../types/index.js';

/**
 * Control Set Validation Gate for LLM-as-a-Judge
 * Prevents corrupted or hallucinated evaluation metrics.
 * 
 * Invariant: If a judge cannot correctly score gold-labeled control queries
 * with at least 90% accuracy, its verdicts are rejected automatically.
 */
export class JudgeValidationGate {
  private readonly thresholdAccuracy: number;

  constructor(thresholdAccuracy = 0.90) {
    this.thresholdAccuracy = thresholdAccuracy;
  }

  /**
   * Pre-flight verification of judge discriminator capability
   */
  public async validateJudge(
    controlItems: JudgeControlItem[],
    judgeFn: (query: string, chunk: string) => Promise<boolean>
  ): Promise<{ isValid: boolean; accuracy: number; passedCount: number; totalCount: number }> {
    if (controlItems.length === 0) {
      return { isValid: false, accuracy: 0, passedCount: 0, totalCount: 0 };
    }

    let passedCount = 0;

    for (const item of controlItems) {
      const predictedGrounded = await judgeFn(item.queryText, item.chunkText);
      if (predictedGrounded === item.expectedGrounded) {
        passedCount++;
      }
    }

    const accuracy = passedCount / controlItems.length;
    const isValid = accuracy >= this.thresholdAccuracy;

    return {
      isValid,
      accuracy,
      passedCount,
      totalCount: controlItems.length,
    };
  }
}
