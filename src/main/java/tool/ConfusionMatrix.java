package tool;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * @author Max Ngai
 * @since 2022/10/28
 */
@Data
@Builder
@AllArgsConstructor
public class ConfusionMatrix {

    private int trueNegative;
    private int falseNegative;
    private int truePositive;
    private int falsePositive;

    public ConfusionMatrix(double[] yHat, double[] y) {
        for (int i = 0; i < yHat.length; i++) {
            if (yHat[i] == 1D) {
                if (y[i] == 1D) {
                    truePositive++;
                } else {
                    falsePositive++;
                }
            } else {
                if (y[i] == -1D) {
                    trueNegative++;
                } else {
                    falseNegative++;
                }
            }
        }
    }

    public ConfusionMatrix() {}

    public void table() {
        System.out.println("TN\t");
    }


}
