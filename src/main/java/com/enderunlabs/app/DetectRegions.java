package com.enderunlabs.app;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

/**
 * Created by Lenovo on 18.08.2016.
 */
public class DetectRegions {

    int count = 1;

    public ArrayList<Plate> segmentation(Mat input) {

        Mat inputTwo = input.clone();
        ArrayList<Plate> output = new ArrayList<Plate>();

        Mat imgGray = new Mat();
        Imgproc.cvtColor(input, imgGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(imgGray, imgGray, new Size(5, 5));

        Mat imgSobel = new Mat();
        Imgproc.Sobel(imgGray, imgSobel, CvType.CV_8U, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);

        Mat imgThreshold = new Mat();
        Imgproc.threshold(imgSobel, imgThreshold, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17, 3));
        Imgproc.morphologyEx(imgThreshold, imgThreshold, Imgproc.MORPH_CLOSE, element);

        ArrayList<RotatedRect> rects = new ArrayList<RotatedRect>();
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(imgThreshold, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        ListIterator<MatOfPoint> itc = contours.listIterator();
        while (itc.hasNext()) {
            MatOfPoint2f mp2f = new MatOfPoint2f(itc.next().toArray());
            RotatedRect mr = Imgproc.minAreaRect(mp2f);
            double area = Math.abs(Imgproc.contourArea(mp2f));

            double bbArea = mr.size.area();
            double ratio = area / bbArea;
            if ((ratio < 0.45) || (bbArea < 1000)) {
                itc.remove();  // other than deliberately making the program slow,
                // does erasing the contour have any purpose?
            } else {
                rects.add(mr);
            }
        }

        Mat result = new Mat();
        input.copyTo(result);

        Imgproc.drawContours(result, contours, -1, new Scalar(255, 0, 0), 1);

        for (int i = 0; i < rects.size(); i++) {
            Imgproc.circle(result, rects.get(i).center, 3, new Scalar(0, 255, 0), -1);
            float minSize = (float) ((rects.get(i).size.width < rects.get(i).size.height) ? rects.get(i).size.width : rects.get(i).size.height);
            minSize = minSize - minSize * 0.5f;

            Random rand = new Random(System.currentTimeMillis());

            Mat mask = new Mat();
            mask.create(input.rows() + 2, input.cols() + 2, CvType.CV_8UC1);
            mask.setTo(Scalar.all(0));

            int loDiff = 30;
            int upDiff = 30;
            int connectivity = 4;
            int newMaskVal = 255;
            int numSeeds = 10;

            Rect ccomp = new Rect();

            int flags = connectivity + (newMaskVal << 8) + Imgproc.FLOODFILL_FIXED_RANGE + Imgproc.FLOODFILL_MASK_ONLY;

            for (int j = 0; j < numSeeds; j++) {
                Point seed = new Point();

                seed.x = rects.get(i).center.x + rand.nextInt((int) (minSize + 1)) - (minSize / 2);
                seed.y = rects.get(i).center.y + rand.nextInt((int) (minSize + 1)) - (minSize / 2);
                Imgproc.circle(result, seed, 1, new Scalar(0, 255, 255), -1);
                int area = Imgproc.floodFill(input, mask, seed, new Scalar(255, 0, 0), ccomp,
                        new Scalar(loDiff, loDiff, loDiff), new Scalar(upDiff, upDiff, upDiff), flags);
            }


            ArrayList<Point> pointsInterestList = new ArrayList<Point>();

            for (int j = 0; j < mask.rows(); j++) {
                for (int k = 0; k < mask.cols(); k++) {
                    double[] pixel = mask.get(j, k);

                    if (pixel[0] == 255) {
                        //add Point of Mat to list of points
                        Point point = new Point(k, j);
                        pointsInterestList.add(point);
                    }
                }
            }

            MatOfPoint2f m2fFromList = new MatOfPoint2f();
            m2fFromList.fromList(pointsInterestList); //create MatOfPoint2f from list of points
            MatOfPoint2f m2f = new MatOfPoint2f();
            m2fFromList.convertTo(m2f, CvType.CV_32FC2); //convert to type of MatOfPoint2f created from list of points

            RotatedRect minRect = Imgproc.minAreaRect(m2f);

            if (verifySizes(minRect)) {


                Point[] vertices = new Point[4];
                minRect.points(vertices);
                List<MatOfPoint> boxContours = new ArrayList<MatOfPoint>();
                boxContours.add(new MatOfPoint(vertices));
                Imgproc.drawContours(result, boxContours, 0, new Scalar(128, 128, 128), -1);
                float r = (float) minRect.size.width / (float) minRect.size.height;
                float angle = (float) minRect.angle;
                if (r < 1)
                    angle = 90 + angle;
                Mat rotmat = Imgproc.getRotationMatrix2D(minRect.center, angle, 1);
                Mat imgRotated = new Mat();
                Imgproc.warpAffine(input, imgRotated, rotmat, input.size(), Imgproc.INTER_CUBIC);
                Size rectSize = minRect.size;
                if (r < 1)
                    swap(rectSize.width, rectSize.height);
                Mat imgCrop = new Mat();
                Imgproc.getRectSubPix(imgRotated, rectSize, minRect.center, imgCrop);
                Mat resultResized = new Mat();
                resultResized.create(33, 144, CvType.CV_8UC3);
                Imgproc.resize(imgCrop, resultResized, resultResized.size(), 0, 0, Imgproc.INTER_CUBIC);
                Mat grayResult = new Mat();
                Imgproc.cvtColor(resultResized, grayResult, Imgproc.COLOR_BGR2GRAY);
                Imgproc.blur(grayResult, grayResult, new Size(3, 3));
                grayResult = histeq(grayResult);
                Imgcodecs.imwrite("data/out_" + count + ".png", grayResult);
                output.add(new Plate(grayResult, minRect.boundingRect()));
                count++;
            }
        }
        return output;
    }

    public static boolean verifySizes(RotatedRect candidate) {
        double error = 0.4;
        final double aspect = 4.7272; // Turkey license plate aspect ratio
        double min = 15 * aspect * 15;
        double max = 125 * aspect * 125;
        double rmin = aspect - aspect * error;
        double rmax = aspect + aspect * error;
        double area = candidate.size.height * candidate.size.width;
        double r = candidate.size.width / candidate.size.height;
        if (r < 1)
            r = candidate.size.height / candidate.size.width;
        if ((area < min || area > max) || (r < rmin || r > rmax)) {
            return false;
        } else {
            return true;
        }
    }

    public static Mat histeq(Mat input) {
        Mat output = new Mat(input.size(), input.type());
        if (input.channels() == 3) {
            Mat hsv = new Mat();
            ArrayList<Mat> hsvSplit = new ArrayList<Mat>();
            Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
            Core.split(hsv, hsvSplit);
            Imgproc.equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
            Core.merge(hsvSplit, hsv);
            Imgproc.cvtColor(hsv, output, Imgproc.COLOR_HSV2BGR);
        } else if (input.channels() == 1) {
            Imgproc.equalizeHist(input, output);
        }
        return output;
    }

    public static void swap(double a, double b) {
        double temp = a;
        a = b;
        b = temp;
    }

    public ArrayList<Plate> run(Mat input) {
        ArrayList<Plate> tmp = segmentation(input);
        return tmp;
    }

    public static boolean verifySizesTwo(RotatedRect mr, MatOfPoint2f mp2f) {

        double area = Math.abs(Imgproc.contourArea(mp2f));

        double bbArea = mr.size.area();
        double ratio = area / bbArea;

        if ((ratio < 0.45) || (bbArea < 400)) {
            return false;
        } else {
            return true;
        }
    }

}
