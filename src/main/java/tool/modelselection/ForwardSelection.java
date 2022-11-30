package tool.modelselection;

import org.apache.commons.math3.linear.RealMatrix;
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
        train();
    }

    private void train() {
        TreeSet<Integer> rest = new TreeSet<>(IntStream.range(0, p).boxed().collect(Collectors.toList()));
        List<PredictorCombo> candidate = new ArrayList<>();
        for (Integer z = 0; z < maxPredictors; z++) {
            try {

            PredictorCombo minRss = rest.parallelStream()
                    .map(e -> {
                        PredictorCombo predictors = candidate.size() == 0
                                ? new PredictorCombo()
                                : new PredictorCombo(candidate.get(candidate.size() - 1));
                        predictors.add(e);
                        return predictors;
                    })
                    .sorted(Comparator.comparingDouble(e -> model.train(getXByPredictors(e), Y).trainRss()))
                    .findFirst()
                    .get();

            candidate.add(minRss);
            rest.removeAll(minRss.getSelected());
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

        List<Score> res = candidate.parallelStream()
                .map(e -> {
                    Score score = new Score();
                    score.setK(e.getSize());
                    score.setSelectedX(e);
                    score.setRss(CrossValidation.kFoldCv(getXByPredictors(e), Y, model, kFold) * n);
                    return score;
                })
                .sorted(Comparator.comparingDouble(Score::getRss))
                .collect(Collectors.toList());

        this.res = res;
    }
}
