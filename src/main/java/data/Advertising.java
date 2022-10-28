package data;

import com.google.common.base.Function;
import com.google.common.base.Supplier;
import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Max Ngai
 * @since 2022/10/9
 */
@Data
public class Advertising {

    private Array2DRowRealMatrix advertising;

    public Advertising() {
        advertising = DataReader.read("Advertising");

    }

    public double[] getY() {
        return advertising.getColumn(4);
    }

    public double[] getTV() {
        return advertising.getColumn(1);

    }

    public double[] getRadio() {
        return advertising.getColumn(2);


    }

    public double[] getNewspaper() {
        return advertising.getColumn(3);

    }

    public double[][] getXMatrix() {
        RealMatrix sub = advertising.getSubMatrix(0, advertising.getRowDimension() - 1, 1, 3);
        return sub.getData();
    }

    public double[][] getTvXRadio() {
        RealVector tv = advertising.getColumnVector(1);
        RealVector radio = advertising.getColumnVector(2);
        RealVector realVector = tv.ebeMultiply(radio);
        Array2DRowRealMatrix res = new Array2DRowRealMatrix(tv.getDimension(), 3);
        res.setColumnVector(0, tv);
        res.setColumnVector(1, radio);
        res.setColumnVector(2, realVector);
        return res.getData();
    }






}
