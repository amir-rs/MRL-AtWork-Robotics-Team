/**
 * @file /include/randomGenerator_GUI/main_window.hpp
 *
 * @brief Qt based gui for randomGenerator_GUI.
 *
 * @date November 2010
 **/
#ifndef randomGenerator_GUI_MAIN_WINDOW_H
#define randomGenerator_GUI_MAIN_WINDOW_H

/*****************************************************************************
** Includes
*****************************************************************************/

#include <QtGui/QMainWindow>
#include "ui_main_window.h"
#include "qnode.hpp"
#include <ros/package.h>
#include "randomGenerator/randomGenerator.hpp"

/*****************************************************************************
** Namespace
*****************************************************************************/

namespace randomGenerator_GUI {

/*****************************************************************************
** Interface [MainWindow]
*****************************************************************************/
/**
 * @brief Qt central, all operations relating to the view part here.
 */
class MainWindow : public QMainWindow {
Q_OBJECT

public:
	MainWindow(int argc, char** argv, QWidget *parent = 0);
	~MainWindow();

	void closeEvent(QCloseEvent *event); // Overloaded function

public Q_SLOTS:
	/******************************************
	** Auto-connections (connectSlotsByName())
	*******************************************/


    /******************************************
    ** Manual connections
    *******************************************/


private slots:
    // Action when Button CheckAll ist clicked. All items in the listwidget will be checked
    void on_pushButton_CheckAll_clicked();

    // Action when an item in the listwidget has been clicked. Depending on the choosen value of
    // comboBox_complexity all items ("High"), up to 6 items are checkable ("Medium"), up to 4 items
    // are checkable ("Low") in the listwidget,
    void on_listWidget_itemClicked(QListWidgetItem *item);

    // Action when the value of string has changed. Following actions are done depending on the choosen argument:
    // High     --> Check all items in listwidget
    // Medium   --> Uncheck all items in listwidget
    // Low      --> Uncheck all items in listwidget
    void on_comboBox_complexity_currentIndexChanged(const QString &arg1);

    // Action when Button Uncheck All is clicked. Will uncheck all items in listwidget
    void on_pushButton_UncheckAll_clicked();

    // Action when Button Generate is clicked. The tableWidget will be set with random items
    void on_pushButton_Generate_clicked();

    // Action when Button Delete is clicked. All cells in the tableWidget will be cleared
    void on_pushButton_Delete_clicked();

    void on_pushButton_Save_clicked();

    void on_comboBox_Test_currentIndexChanged(const QString &arg1);

    void on_spinBox_valueChanged(int arg1);

private:
	Ui::MainWindowDesign ui;
	QNode qnode;
    randomGenerator *rand = new(randomGenerator);
};

}  // namespace randomGenerator_GUI

#endif // randomGenerator_GUI_MAIN_WINDOW_H
