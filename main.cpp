#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Lapangan
int hminLapangan = 70, sminLapangan = 186, vminLapangan = 146;
int hmaxLapangan = 122, smaxLapangan = 255, vmaxLapangan = 255;

// Team A
int hminTeamA = 56, sminTeamA = 13, vminTeamA = 73;
int hmaxTeamA = 139, smaxTeamA = 181, vmaxTeamA = 191;

// Ball
int hminBall = 0, sminBall = 162, vminBall = 125;
int hmaxBall = 51, smaxBall = 255, vmaxBall = 255;

// Line
int hminLine = 0, sminLine = 149, vminLine = 0;
int hmaxLine = 255, smaxLine = 255, vmaxLine = 255;

static const double thresholdAreaTimA = 150.0;
static const double thresholdAreaBall = 150.0;

int main(int, char **)
{
    string path = "../video/video3.mp4";
    VideoCapture cap(path);
    Mat frame, frameHSV, frameBlur, frameHSL;
    Mat maskLapangan, maskTeamA, kernel, maskBall, maskLine;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }

        cvtColor(frame, frameHSV, COLOR_BGR2HSV);
        cvtColor(frame, frameHSL, COLOR_BGR2HLS);
        blur(frameHSV, frameBlur, Size(3, 3));

        inRange(frameBlur, Scalar(hminLapangan, sminLapangan, vminLapangan), Scalar(hmaxLapangan, smaxLapangan, vmaxLapangan), maskLapangan);
        inRange(frameBlur, Scalar(hminTeamA, sminTeamA, vminTeamA), Scalar(hmaxTeamA, smaxTeamA, vmaxTeamA), maskTeamA);
        inRange(frameBlur, Scalar(hminBall, sminBall, vminBall), Scalar(hmaxBall, smaxBall, vmaxBall), maskBall);
        inRange(frameHSL, Scalar(hminLine, sminLine, vminLine), Scalar(hmaxLine, smaxLine, vmaxLine), maskLine);

        kernel = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
        morphologyEx(maskLapangan, maskLapangan, MORPH_CLOSE, kernel);

        vector<vector<Point>> contoursLapangan, contoursBall, contoursTeamA, contoursLine;
        vector<Vec4i> hierarchyLapangan, hierarchyBall, hierarchyTeamA, hierarchyLine;
        findContours(maskLapangan, contoursLapangan, hierarchyLapangan, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskTeamA, contoursTeamA, hierarchyTeamA, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskBall, contoursBall, hierarchyBall, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> boundingRects;
        Rect largestRectLapangan;
        int largestIndex = -1;
        double maxArea = 0;

        for (size_t i = 0; i < contoursLapangan.size(); i++)
        {
            double area = contourArea(contoursLapangan[i]);
            if (area > thresholdAreaTimA)
            {
                Rect boundingRectLapangan = boundingRect(contoursLapangan[i]);
                bool merged = false;

                for (auto &rect : boundingRects)
                {
                    if ((abs(boundingRectLapangan.x - rect.x) < 1200) && (abs(boundingRectLapangan.y - rect.y) < 1200))
                    {
                        rect |= boundingRectLapangan;
                        merged = true;
                        break;
                    }
                }

                if (!merged)
                {
                    boundingRects.push_back(boundingRectLapangan);
                }

                if (area > maxArea)
                {
                    maxArea = area;
                    largestRectLapangan = boundingRectLapangan;
                    largestIndex = i;
                }
            }
        }

        for (const auto &rect : boundingRects)
        {
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
        }

        for (size_t k = 0; k < contoursBall.size(); k++)
        {
            double areaBall = contourArea(contoursBall[k]);
            if (areaBall < thresholdAreaBall)
            {
                continue;
            }

            Point2f center;
            float radius;
            minEnclosingCircle(contoursBall[k], center, radius);
            bool isBallInsideLapangan = false;

            if (largestIndex != -1)
            {
                if (largestRectLapangan.contains(center))
                {
                    isBallInsideLapangan = true;
                }
            }

            circle(frame, center, (int)radius, Scalar(0, 255, 0), 2);
            if (isBallInsideLapangan)
            {
                putText(frame, "Bola di dalam lapangan", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            }
            else
            {
                putText(frame, "Bola di luar lapangan", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            }
        }

        imshow("Frame", frame);
        imshow("Mask Lapangan", maskLapangan);

        if (waitKey(18) == 27)
            break;
    }

    return 0;
}