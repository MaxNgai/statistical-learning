package tool.modelselection;

import lombok.Data;

/**
 *
 * performance score of a model
 *
 * @author Max Ngai
 * @since 2022/11/30
 */
@Data
public class Score {
    /**
     * number of selected predictors
     */
    private int k;

    /**
     * column index of selected predictors,
     */
    private PredictorCombo selectedX;


    private double rss;
}
