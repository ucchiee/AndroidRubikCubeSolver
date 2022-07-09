package jp.ac.titech.itpro.sdl.rubikcubesolver;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.ml.Ml.ROW_SAMPLE;

import org.opencv.utils.Converters;

import static java.lang.Math.max;
import static java.lang.Math.min;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

// import android.Manifest;
// import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.widget.ImageView;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.TrainData;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    /*** Fixed values ***/
    private static final String TAG = "RubikCubeSolver";
    final private int REQUEST_CODE_FOR_PERMISSIONS = 1234;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};

    /*** Views ***/
    private PreviewView previewView;
    private ImageView imageView;
    /*** For CameraX ***/
    private Camera camera = null;
    private Preview preview = null;
    private ImageAnalysis imageAnalysis = null;
    final private ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    /*** For Rubik's Cube Solver ***/
    private Mat trainData = new Mat(6, 4, CV_32F);
    private KNearest knn = KNearest.create();

    static {
        System.loadLibrary("opencv_java4");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);

        if (checkPermissions()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_FOR_PERMISSIONS);
        }

        // prepare K-nearest neighbor
        for (int i = 0; i < 6; i++) {
            trainData.put(i, 0, Util.colorData[i]);
        }
        knn.train(trainData, ROW_SAMPLE, Converters.vector_int_to_Mat(Util.colorResponse));
    }

    private void startCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        Context context = this;
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    preview = new Preview.Builder().build();
                    imageAnalysis = new ImageAnalysis.Builder().build();
                    imageAnalysis.setAnalyzer(cameraExecutor, new MyImageAnalyzer());
                    CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

                    cameraProvider.unbindAll();
                    camera = cameraProvider.bindToLifecycle((LifecycleOwner) context, cameraSelector, preview, imageAnalysis);
                    preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));
                } catch (Exception e) {
                    Log.e(TAG, "[startCamera] Use case binding failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private class MyImageAnalyzer implements ImageAnalysis.Analyzer {
        private Mat matPrevious = null;
        private String[][] detectedColor = {{null, null, null}, {null, null, null}, {null, null, null}};

        @Override
        public void analyze(@NonNull ImageProxy image) {
            /* Create cv::mat(RGB888) from image(NV21) */
            Mat matOrg = getMatFromImage(image);

            /* Fix image rotation (it looks image in PreviewView is automatically fixed by CameraX???) */
            Mat mat = fixMatRotation(matOrg);

            // Log.i(TAG, "[analyze] width = " + image.getWidth() + ", height = " + image.getHeight() + "Rotation = " + previewView.getDisplay().getRotation());
            // Log.i(TAG, "[analyze] mat width = " + matOrg.cols() + ", mat height = " + matOrg.rows());

            /* Do some image processing */
            Mat matOutput = new Mat(mat.rows(), mat.cols(), mat.type());
            mat.copyTo(matOutput);
            // if (matPrevious == null) matPrevious = mat;
            // Core.absdiff(mat, matPrevious, matOutput);
            // matPrevious = mat;

            // calculate each point
            double cubeLen = min(image.getWidth(), image.getHeight()) * 0.8;
            int boxLen = (int) (cubeLen / 3);
            int startX = (int) ((min(image.getWidth(), image.getHeight()) - cubeLen) / 2);
            int startY = (int) (max(image.getHeight(), image.getWidth()) * 0.2);

            // calc average RGB of each box
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Mat color = Util.calcBoxColorAve(mat, startX + boxLen * i, startY + boxLen * j, boxLen);
                    Mat res = new Mat();
                    knn.findNearest(color, 1, res);
                    detectedColor[i][j] = Util.colorLabel[(int)res.get(0, 0)[0]];
                }
            }

            // draw frame and detected color
            drawCubeFrame(matOutput, startX, startY, boxLen, new Scalar(255, 0, 0), 2);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Imgproc.putText(matOutput, detectedColor[i][j], new Point(startX + boxLen * i, startY + boxLen * j), 2, 3, new Scalar(255, 0, 0));
                    // Log.i(TAG, "[analyze] (" + i + ", " + j + ") = " + detectedColor[i][j].dump());
                }
            }

            /* Convert cv::mat to bitmap for drawing */
            Bitmap bitmap = Bitmap.createBitmap(matOutput.cols(), matOutput.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matOutput, bitmap);

            /* Display the result onto ImageView */
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    imageView.setImageBitmap(bitmap);
                }
            });

            /* Close the image otherwise, this function is not called next time */
            image.close();
        }

        private void drawCubeFrame(Mat mat, int startX, int startY, int boxLen, Scalar scalar, int thickness) {
            /* Draw 2d flat cube */
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Imgproc.rectangle(mat, new Rect(startX + boxLen * i, startY + boxLen * j, boxLen, boxLen), scalar, thickness);
                }
            }
        }

        private Mat getMatFromImage(ImageProxy image) {
            /* Create cv::mat(RGB888) from image(NV21) */
            /* https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb */
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuv.put(0, 0, nv21);
            Mat mat = new Mat();
            Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2RGB_NV21, 3);
            return mat;
        }

        private Mat fixMatRotation(Mat matOrg) {
            Mat mat;
            switch (previewView.getDisplay().getRotation()) {
                default:
                case Surface.ROTATION_0:
                    mat = new Mat(matOrg.cols(), matOrg.rows(), matOrg.type());
                    Core.transpose(matOrg, mat);
                    Core.flip(mat, mat, 1);
                    break;
                case Surface.ROTATION_90:
                    mat = matOrg;
                    break;
                case Surface.ROTATION_270:
                    mat = matOrg;
                    Core.flip(mat, mat, -1);
                    break;
            }
            return mat;
        }
    }

    private boolean checkPermissions() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_FOR_PERMISSIONS) {
            if (checkPermissions()) {
                startCamera();
            } else {
                Log.i(TAG, "[onRequestPermissionsResult] Failed to get permissions");
                this.finish();
            }
        }
    }
}