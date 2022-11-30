package tool.modelselection;

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
        train();
    }

    private void train() {
        List<Integer> ALL = IntStream.range(0, p).boxed().collect(Collectors.toList());
        TreeSet<Integer> within = new TreeSet<>(ALL);
        List<PredictorCombo> candidate = new ArrayList<>();
        candidate.add(new PredictorCombo(ALL));
        for (Integer z = 0; z < p - 1; z++) {
            PredictorCombo minRss = within.parallelStream()
                    .map(e -> {
                        PredictorCombo predictors = new PredictorCombo(candidate.get(candidate.size() - 1));
                        predictors.remove(e);
                        return predictors;
                    })
                    .sorted(Comparator.comparingDouble(e -> model.train(getXByPredictors(e), Y).trainRss()))
                    .findFirst()
                    .get();

            candidate.add(minRss);
            within.retainAll(minRss.getSelected());
        }

        List<PredictorCombo> candidate0 = candidate.stream()
                .filter(e -> e.getSize() <= maxPredictors).collect(Collectors.toList());

        List<Score> res = candidate0.parallelStream()
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
