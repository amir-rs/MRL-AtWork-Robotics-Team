/** *****************************************************************
 *
 * Copyright (c) 2012,
 * Institute of Mechatronic Systems,
 * Leibniz Universitaet Hannover.
 * (BSD License)
 * All rights reserved.
 *
 * http://www.imes.uni-hannover.de
 *
 * This software is distributed WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.
 *
 * For further information see http://www.linfo.org/bsdlicense.html
 *
 ******************************************************************/
/**
 * @file   randomGenerator.cpp
 * @author Eduard Popp <eduardpopp@web.de>
 * @date   25.April 2013
 *
 * @brief  This file contains a randomGenerator for the referee box
 */

#include <randomGenerator/randomGenerator.hpp>
#include <time.h>

using namespace std;

randomGenerator::randomGenerator() {

    objectList.resize(0);
    tagList.resize(0);
    directionlist.clear();
    orientation.clear();
    position.clear();


    // Orientierungen
    for (int angle = 0; angle <= 180; angle += 45) {
        stringstream ss;
        ss << angle;
        orientation.push_back(ss.str());
    }

    position.push_back("upright");
    position.push_back("lying");

    directionlist.push_back("north");
    directionlist.push_back("east");
    directionlist.push_back("south");
    directionlist.push_back("west");
}

randomGenerator::~randomGenerator() { }

void randomGenerator::init() {
    objectList.resize(0);
    tagList.resize(0);
    directionlist.clear();
    orientation.clear();
    position.clear();

    // Orientierungen
    for (int angle = 0; angle <= 180; angle += 45) {
        stringstream ss;
        ss << angle;
        orientation.push_back(ss.str());
    }

    position.push_back("upright");
    position.push_back("lying");

    directionlist.push_back("north");
    directionlist.push_back("east");
    directionlist.push_back("south");
    directionlist.push_back("west");

}


void randomGenerator::addItem(string item) {
    objectList.push_back(item);
}

void randomGenerator::addTag(string tag) {
    tagList.push_back(tag);
}


int randomGenerator::getRandomNumber(int randomSet) {

    if (randomSet <= 0)
        return -1;
    else {
        int randomNumber = rand()% randomSet;
        return randomNumber;
    }
}

bool randomGenerator::getRandObjectList(string &rand_orientation, string &rand_object, string &rand_position) {
    if ((int)objectList.size() > 0) {
        int temp_orientation = getRandomNumber(orientation.size());
        int temp_object = getRandomNumber(objectList.size());
        int temp_position = getRandomNumber(position.size());

        rand_object = objectList.at(temp_object);
        rand_orientation = orientation.at(temp_orientation);
        rand_position = position.at(temp_position);

        objectList.erase(objectList.begin()+ temp_object);
        return true;

    }
    else
        return false;
}


bool randomGenerator::getRandTagList(string &rand_direction, string &rand_tag) {
    if ((int)tagList.size() > 0) {
        int temp_direction = getRandomNumber(directionlist.size());
        int temp_tag = getRandomNumber(tagList.size());

        rand_direction = directionlist.at(temp_direction);
        rand_tag = tagList.at(temp_tag);

        tagList.erase(tagList.begin()+ temp_tag);
        return true;
    }
    else
        return false;
}

int main (){


    randomGenerator randomGen;
    srand(time(NULL));
    vector <int> orientation;
    vector <string> objects;
    int rand_Orientation;
    int rand_Object;
    string current_Object;

    // Orientierungen
    for (int angle = 0; angle <= 180; angle += 45) {
        orientation.push_back(angle);
        cout << "Angle: " << orientation.back() << endl;
    }

    randomGen.addItem("string");



    for (int i=0; i < 10; i++) {
        if (objects.size() >= 1) {
            rand_Orientation = randomGen.getRandomNumber(orientation.size());
            rand_Object = randomGen.getRandomNumber(objects.size());
            current_Object = objects.at(rand_Object);
            cout << i+1 << ".Angle: " << orientation.at(rand_Orientation) << "\t Object: " << current_Object << endl;
            objects.erase(objects.begin()+rand_Object);
            cout << "********" << endl;
        }
        else {
            cout << "---Speicherzugriffsfehler---" << endl;
            return 0;
        }

    }

    return 0;

}
