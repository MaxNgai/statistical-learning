package data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Max Ngai
 * @since 2022/11/2
 */
public class Caravan {
    private Array2DRowRealMatrix data;

    public Caravan() {
        Array2DRowRealMatrix d = DataReader.read("Caravan", new CaravanDataParser());
        data = d;
    }

    public double[][] getX() {
        RealMatrix m = data.getSubMatrix(0, 5821, 0, 84);
        return m.getData();
    }

    public double[] getY() {
        return data.getColumn(85);
    }

    private static class CaravanDataParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            switch (raw) {
                case "Yes": return 1D;
                case "No": return -1D;
                default: return Double.parseDouble(raw);
            }
        }
    }
}
