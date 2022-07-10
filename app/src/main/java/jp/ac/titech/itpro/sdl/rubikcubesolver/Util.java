package jp.ac.titech.itpro.sdl.rubikcubesolver;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;


import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Util {
    private static final String TAG = "RubikCubeSolverUtil";

    static final double[][] colorData = new double[][]{
            new double[]{255, 0, 0, 0},  // R
            new double[]{0, 255, 0, 0},  // G
            new double[]{0, 0, 255, 0},  // B
            new double[]{255, 255, 255, 0},  // W
            new double[]{255, 255, 0, 0},  // Y
            new double[]{255, 165, 0, 0}  // O
    };
    static final String[] colorLabel = new String[]{"R", "G", "B", "W", "Y", "O"};
    static final List<Integer> colorResponse = new ArrayList<>(Arrays.asList(0, 1, 2, 3, 4, 5));

    static Mat calcBoxColorAve(Mat mat, int boxX, int boxY, int boxLen) {
        // extract box as sub matrix
        Mat boxMat = mat.submat(new Rect(boxX, boxY, boxLen, boxLen));

        // create mask
        Mat mask = new Mat(boxMat.cols(), boxMat.rows(), CV_8U);
        mask.setTo( new Scalar( 0.0 ) );
        int innerBoxLen = (int) (boxLen * 0.6);
        int innerX = (int) ((boxLen - innerBoxLen) / 2);
        int innerY = (int) ((boxLen - innerBoxLen) / 2);
        Imgproc.rectangle(mask, new Rect(innerX, innerY, innerBoxLen, innerBoxLen), new Scalar(255, 255, 255), -1);

        Scalar mean = Core.mean(boxMat, mask);
        Mat ret = new Mat(1, mean.val.length, CV_32F);
        for (int i = 0; i < 4; i++) {
            ret.put(0, i, mean.val[i]);
        }
        Log.v(TAG, "mean(matrix) : " + ret.dump());
        return ret;
    }
}
