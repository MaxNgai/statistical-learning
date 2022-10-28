package data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import tool.Macro;

/**
 * @author Max Ngai
 * @since 2022/10/18
 */
public class Default {

    private Array2DRowRealMatrix data;

    public Default() {
        Array2DRowRealMatrix d = DataReader.read("Default", new DefaultDataParser());
        data = d;
    }

    public double[] getDefault() {
        return data.getColumn(0);
    }

    public double[] getStudent() {
        return data. getColumn(1);
    }

    public double[] getBalance() {
        return data.getColumn(2);
    }

    public double[] getIncome() {
        return new ArrayRealVector(data.getColumn(3)).mapDivide(1).toArray();
    }

    public double[][] getX() {
        return Macro.matrixHConcat(getStudent(), getBalance(), getIncome()).getData();
    }

    public double[][] getStudentAndBalance() {
        return Macro.matrixHConcat(getStudent(), getBalance()).getData();
    }

    private static class DefaultDataParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            switch (raw) {
                case "No": return 0D;
                case "Yes": return 1D;
                default: return Double.parseDouble(raw);
            }
        }
    }
}
