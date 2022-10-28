package tool;

import lombok.Builder;
import lombok.Data;

/**
 * @author Max Ngai
 * @since 2022/10/28
 */
@Data
@Builder
public class ConfusionMatrix {

    private int trueNegative;
    private int falseNegative;
    private int truePositive;
    private int falsePositive;

}
