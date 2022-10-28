import data.Default;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import tool.Covariance;

import java.util.Arrays;

/**
 * @author Max Ngai
 * @since 2022/9/20
 */
public class Application {


    public static void main(String[] args) {

        ArrayRealVector first = new ArrayRealVector(new double[]{1, 2,3, 4});
        ArrayRealVector second = new ArrayRealVector(new double[]{5, 6, 7, 8});

        RealMatrix m1 = new Array2DRowRealMatrix(new double[][]{new double[]{1, 2}, new double[]{3, 4}});
        RealMatrix m2 = new Array2DRowRealMatrix(new double[][]{new double[]{5, 6}, new double[]{7, 8}});

        System.out.println(first.outerProduct(second)); // 外积是1向量为竖着复制，2向量为横着复制，对应位置的两复制数的乘积
        m1.multiply(m2); // 矩阵乘积，右乘。m1是左边
        m2.preMultiply(m1); // 矩阵左乘，m2是右边


    }
}
