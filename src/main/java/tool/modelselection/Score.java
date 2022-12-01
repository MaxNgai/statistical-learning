package tool.modelselection;

import lombok.AllArgsConstructor;
import lombok.Data;
import tool.model.Model;

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
     * column index of selected predictors,
     */
    private PredictorCombo selectedX;

    private Model model;

    private double testRss;

    public Score(PredictorCombo selectedX, Model model) {
        this.selectedX = selectedX;
        this.model = model;
    }

    public double getTrainRss() {
        return model.trainRss();
    }

}
