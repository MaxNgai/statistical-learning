package tool.modelselection;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import tool.CrossValidation;
import tool.model.Model;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/30
 */
public class ForwardSelection extends ModelSelection {

    public ForwardSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        super(X, Y, model, maxPredictors);
        cacheScoreTrainedByAllData = regSubset(this.X, this.Y);
    }

    @Override
    protected List<Score> regSubset(RealMatrix input, RealVector output) {
        int[] selectedRows = IntStream.range(0, input.getRowDimension()).toArray();
        TreeSet<Integer> rest = new TreeSet<>(IntStream.range(0, p).boxed().collect(Collectors.toList()));
        List<Score> candidate = new ArrayList<>();
        for (Integer z = 0; z < maxPredictors; z++) {
            Score score = rest.parallelStream()
                    .map(e -> {
                        PredictorCombo predictors = candidate.size() == 0
                                ? new PredictorCombo()
                                : new PredictorCombo(candidate.get(candidate.size() - 1).getSelectedX());
                        predictors.add(e);
                        int[] columns = predictors.getArray();
                        RealMatrix subMatrix = input.getSubMatrix(selectedRows, columns);
                        Model train = model.train(subMatrix, output);
                        return new Score(predictors, train);
                    })
                    .sorted(Comparator.comparingDouble(Score::getTrainRss))
                    .findFirst()
                    .get();

            candidate.add(score);
            rest.removeAll(score.getSelectedX().getSelected());
        }

        return candidate;
    }



}
