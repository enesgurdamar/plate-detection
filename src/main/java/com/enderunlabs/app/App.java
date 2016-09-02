package com.enderunlabs.app;

import com.enderunlabs.app.utils.ImageProcessor;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.ml.Ml;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;

import java.io.File;

import com.enderunlabs.app.utils.ImageProcessor;
import org.xml.sax.SAXException;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.FileStore;
import java.util.*;

public class App extends Thread {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private JFrame frame;
    private JLabel imageLabel;
    private CascadeClassifier carDetector;
    private CascadeClassifier plateDetector;

    public static void main(String[] args) throws InterruptedException, IOException, ParserConfigurationException, SAXException, ParserConfigurationException {

//        App app = new App();
//        app.initGUI();
//        app.runMainLoop();

//        File folder = new File("src/main/resources/images");
//        File[] list = folder.listFiles();
//
//        for (File file : list){
//
//            DetectRegions dr = new DetectRegions();
//            Mat input = Imgcodecs.imread(file.getAbsolutePath());
//
//            ArrayList<Plate> possibleRegions = dr.run(input);
//        }


        DetectRegions dr = new DetectRegions();
        Mat input = Imgcodecs.imread("src/main/resources/images/mrt.png");

        ArrayList<Plate> possibleRegions = dr.run(input);

        File f = new File("src/main/resources/xml/SVM.xml");

        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(f);

        doc.getDocumentElement().normalize();
        NodeList nList = doc.getElementsByTagName("TrainingData");
        NodeList nListTwo = doc.getElementsByTagName("classes");

        Mat SVM_TrainingData = null;
        Mat SVM_Classes = null;

        SVM_TrainingData = fillMat(SVM_TrainingData,nList);
        SVM_Classes = fillMat(SVM_Classes,nListTwo);

        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setDegree(0);
        svm.setGamma(-1);
        svm.setCoef0(0);
        svm.setC(1);
        svm.setNu(0);
        svm.setP(0);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER,1000,0.01));
        // Train SVM
        svm.train(SVM_TrainingData,Ml.ROW_SAMPLE,SVM_Classes);
        ArrayList<Plate> plates = new ArrayList<Plate>();
        for(int i=0; i< possibleRegions.size(); i++){
            Mat img = possibleRegions.get(i).plateImg;
            Mat p = img.reshape(1,1);
            p.convertTo(p,CvType.CV_32FC1);
            int response = (int) svm.predict(p);
            if (response==1)
                plates.add(possibleRegions.get(i));
        }
        System.out.println(plates.size());

    }

    public static Mat fillMat(Mat data, NodeList nList){

        for (int i = 0; i < nList.getLength(); i++) {
            Node node = nList.item(i);

            if (node.getNodeType() == Node.ELEMENT_NODE) {
                Element element = (Element) node;

                String type_id = element.getAttribute("type_id");
                if ("opencv-matrix".equals(type_id) == false) {
                    System.out.println("Fault type_id ");
                }

                String rowsStr = element.getElementsByTagName("rows").item(0).getTextContent();
                String colsStr = element.getElementsByTagName("cols").item(0).getTextContent();
                String dtStr = element.getElementsByTagName("dt").item(0).getTextContent();
                String dataStr = element.getElementsByTagName("data").item(0).getTextContent();

                int rows = Integer.parseInt(rowsStr);
                int cols = Integer.parseInt(colsStr);
                int type = CvType.CV_8U;

                Scanner s = new Scanner(dataStr);
                s.useLocale(Locale.US);

                if ("f".equals(dtStr)) {
                    type = CvType.CV_32F;
                    data = new Mat(rows, cols, type);
                    float fs[] = new float[1];
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            if (s.hasNextFloat()) {
                                fs[0] = s.nextFloat();
                            } else {
                                fs[0] = 0;
                                System.err.println("Unmatched number of float value at rows=" + r + " cols=" + c);
                            }
                            data.put(r, c, fs);
                        }
                    }
                } else if ("i".equals(dtStr)) {
                    type = CvType.CV_32S;
                    data = new Mat(rows, cols, type);
                    int is[] = new int[1];
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            if (s.hasNextInt()) {
                                is[0] = s.nextInt();
                            } else {
                                is[0] = 0;
                                System.err.println("Unmatched number of int value at rows=" + r + " cols=" + c);
                            }
                            data.put(r, c, is);
                        }
                    }
                } else if ("s".equals(dtStr)) {
                    type = CvType.CV_16S;
                    data = new Mat(rows, cols, type);
                    short ss[] = new short[1];
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            if (s.hasNextShort()) {
                                ss[0] = s.nextShort();
                            } else {
                                ss[0] = 0;
                                System.err.println("Unmatched number of int value at rows=" + r + " cols=" + c);
                            }
                            data.put(r, c, ss);
                        }
                    }
                } else if ("b".equals(dtStr)) {
                    data = new Mat(rows, cols, type);
                    byte bs[] = new byte[1];
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            if (s.hasNextByte()) {
                                bs[0] = s.nextByte();
                            } else {
                                bs[0] = 0;
                                System.err.println("Unmatched number of byte value at rows=" + r + " cols=" + c);
                            }
                            data.put(r, c, bs);
                        }
                    }
                }
            }
        }
        return data;
    }

    // Capture a video from a camera
    private void runMainLoop() throws InterruptedException, FileNotFoundException {

        ImageProcessor imageViewer = new ImageProcessor();
        Mat webcamMatImage = new Mat();
        Image tempImage;

        //VideoCapture capture = new VideoCapture(0); // Capture from web cam
        //VideoCapture capture = new VideoCapture("rtsp://192.168.2.64/Streaming/Channels/101"); // Capture from IP Camera
        VideoCapture capture = new VideoCapture("src/main/resources/videos/video2.mp4"); // Capture avi file

        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 800);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 600);

        if (capture.isOpened()) { // Check whether the camera is instantiated
            int i = 1;
            DetectRegions dr = new DetectRegions();
            while (true) { // Retrieve each captured frame in a loop
                capture.read(webcamMatImage);

                if (!webcamMatImage.empty()) {
                    dr.segmentation(webcamMatImage);
                    tempImage = imageViewer.toBufferedImage(webcamMatImage);
                    ImageIcon imageIcon = new ImageIcon(tempImage, "Captured Video");
                    imageLabel.setIcon(imageIcon);
                    frame.pack(); // This will resize the window to fit the image

                } else {
                    System.out.println(" -- Frame not captured or the video has finished! -- Break!");
                    break;
                }
            }
        } else {
            System.out.println("Couldn't open capture.");
        }
    }

    private void initGUI() {

        frame = new JFrame("Camera Input Example");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);
        imageLabel = new JLabel();
        frame.add(imageLabel);
        frame.setVisible(true);
    }

}