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
public class BackwardSelection extends ModelSelection {

    public BackwardSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        super(X, Y, model, maxPredictors);
        cacheScoreTrainedByAllData = regSubset(this.X, this.Y);
    }

    @Override
    protected List<Score> regSubset(RealMatrix input, RealVector output) {
        int[] selectedRows = IntStream.range(0, input.getRowDimension()).toArray();
        List<Integer> ALL = IntStream.range(0, p).boxed().collect(Collectors.toList());
        List<Score> candidate = new ArrayList<>();
        PredictorCombo full = new PredictorCombo(ALL);
        Model fullModel = model.train(input, output);
        candidate.add(new Score(full, fullModel));

        TreeSet<Integer> within = new TreeSet<>(ALL);
        for (Integer z = 0; z < p - 1; z++) {
            Score minRss = within.parallelStream()
                    .map(e -> {
                        PredictorCombo predictors = new PredictorCombo(candidate.get(candidate.size() - 1).getSelectedX());
                        predictors.remove(e);
                        int[] columns = predictors.getArray();
                        RealMatrix subMatrix = input.getSubMatrix(selectedRows, columns);
                        Model train = model.train(subMatrix, output);
                        return new Score(predictors, train);
                    })
                    .sorted(Comparator.comparingDouble(Score::getTrainRss))
                    .findFirst()
                    .get();

            candidate.add(minRss);
            within.retainAll(minRss.getSelectedX().getSelected());
        }

        return candidate;
    }
}
