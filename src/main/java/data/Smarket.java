package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import tool.Macro;

/**
 * @author Max Ngai
 * @since 2022/11/1
 */
@Data
public class Smarket {

    private Array2DRowRealMatrix data;

    public Smarket() {
        Array2DRowRealMatrix d = DataReader.read("Smarket", new SmarketDataParser());
        data = d;
    }

    public double[] getYear() {
        return data.getColumn(0);
    }

    public double[] getLag1() { return data.getColumn(1); }
    public double[] getLag2() { return data.getColumn(2); }
    public double[] getLag3() { return data.getColumn(3); }
    public double[] getLag4() { return data.getColumn(4); }
    public double[] getLag5() { return data.getColumn(5); }
    public double[] getVolume() { return data.getColumn(6); }
    public double[] getToday() { return data.getColumn(7); }
    public double[] getDirection() { return data.getColumn(8); }

    public double[][] getBesidesDirection() {
        return Macro.matrixHConcat(
                getYear(),
                getLag1(),
                getLag2(),
                getLag3(),
                getLag4(),
                getLag5(),
                getVolume(),
                getToday()
        ).getData();
    }


    private static class SmarketDataParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            switch (raw) {
                case "Up": return 1D;
                case "Down": return -1D;
                default: return Double.parseDouble(raw);
            }
        }
    }
}
