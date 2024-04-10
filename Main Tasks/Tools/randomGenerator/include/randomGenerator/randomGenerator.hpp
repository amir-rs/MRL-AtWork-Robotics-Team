/** *****************************************************************
 *
 * Copyright (c) 2013,
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
 * @file   randomGenerator.hpp
 * @author Eduard Popp <eduardpopp@web.de>
 * @date   25.April 2013
 *
 * @brief  This file contains a randomGenerator for the referee box.
 *
 */

#ifndef RANDOMGENERATOR_HPP
#define RANDOMGENERATOR_HPP

#include "ros/ros.h"


class randomGenerator {
public:

    /**
     * @brief Constructor
     */
    randomGenerator();


    /**
     * \brief Destructor
     */
    ~randomGenerator();


    /**
     * @brief clears reintializes the variables #objectList, #orientation and #position
     */
    void init();


    /**
     * @brief Returns a random Number between zero and the value of poolSize
     * @param poolSize
     * @return random number
     */
    int getRandomNumber(int poolSize);


    /**
     * @brief Adds an object to the objectList
     * @param item contains the name of the object adding to the objectList
     */
    void addItem(std::string item);


    void addTag(std::string tag);


    /**
     * @brief returns a random orientation, object and position from the pool. Consider that the choosen object will
     * will be erased from #objectList.
     * @param rand_orientation
     * @param rand_object
     * @param rand_position
     * @return if objectList is empty <true> will be returned, otherwise <false>.
     */
    bool getRandObjectList(std::string &rand_orientation, std::string &rand_object, std::string &rand_position);


    bool getRandTagList(std::string &rand_direction, std::string &rand_tag);



private:
    /**
     * @brief A list containing all objects which are choosen in #getRandObject.
     */
    std::vector <std::string> objectList;


    /**
     * @brief A list containing all orientation between 0° and 180° with 45°step width.
     */
    std::vector <std::string> orientation;


    /**
     * @brief A list containing upright and lying.
     */
    std::vector <std::string> position;


    /**
     * @brief A list containing north,east,west,south.
     */
    std::vector <std::string> directionlist;


    std::vector <std::string> tagList;



};
#endif // RANDOMGENERATOR_HPP
