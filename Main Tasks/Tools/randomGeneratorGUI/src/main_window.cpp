/**
 * @file /src/main_window.cpp
 *
 * @brief Implementation for the qt gui.
 *
 * @date February 2011
 **/
/*****************************************************************************
** Includes
*****************************************************************************/

#include <QtGui>
#include <QMessageBox>
#include <iostream>
#include "../include/randomGenerator_GUI/main_window.hpp"
#include <fstream>

/*****************************************************************************
** Namespaces
*****************************************************************************/

#define MEDIUM  5
#define LOW     3

namespace randomGenerator_GUI {

using namespace Qt;
using namespace std;

/*****************************************************************************
** Implementation [MainWindow]
*****************************************************************************/

MainWindow::MainWindow(int argc, char** argv, QWidget *parent)
	: QMainWindow(parent)
	, qnode(argc,argv)
{
	ui.setupUi(this); // Calling this incidentally connects all ui's triggers to on_...() callbacks in this class.

	// Change the app icon
   	setWindowIcon(QIcon(":/images/icon.png"));
    QObject::connect(&qnode, SIGNAL(rosShutdown()), this, SLOT(close()));

    std::string path;
    if(!ros::package::getPath("randomGenerator_GUI").empty())
        path = ros::package::getPath("randomGenerator_GUI");
    else
        path = "..";

    std::stringstream ss_objectlist;
    ss_objectlist << path << "/objectList.txt";
//    ss_objectlist << "../objectList.txt";
    std::string file_objectlist;
    ss_objectlist >> file_objectlist;
    char * buffer_objectlist = new char[file_objectlist.length()];
    strcpy(buffer_objectlist,file_objectlist.c_str());
    std::fstream f_objectlist;
    f_objectlist.open(buffer_objectlist, ios::in);
    if(f_objectlist) {
        string objectN;
        while(!f_objectlist.eof()) {
            f_objectlist >> objectN;
            QListWidgetItem *item = new(QListWidgetItem);
            item->setCheckState(Unchecked);
            item->setText(objectN.c_str());
            item->setFlags(ItemIsSelectable|ItemIsDragEnabled|ItemIsUserCheckable|ItemIsEnabled);
            ui.listWidget->addItem(item);
        }
    }

}

MainWindow::~MainWindow() {}

/*****************************************************************************
** Implementation [Slots]
*****************************************************************************/

/*****************************************************************************
** Implemenation [Slots][manually connected]
*****************************************************************************/

/*****************************************************************************
** Implementation [Menu]
*****************************************************************************/

/*****************************************************************************
** Implementation [Configuration]
*****************************************************************************/
void MainWindow::closeEvent(QCloseEvent *event)
{
	// Do something upon closing
	QMainWindow::closeEvent(event);
}

}  // namespace randomGenerator_GUI


void randomGenerator_GUI::MainWindow::on_pushButton_CheckAll_clicked()
{
    int len = ui.listWidget->count();
    for (int i = 0;i < len;i++)
        ui.listWidget->item(i)->setCheckState(Checked);
}

void randomGenerator_GUI::MainWindow::on_listWidget_itemClicked(QListWidgetItem *item)
{
    if (ui.comboBox_complexity->currentText().compare("Medium") == 0) {
        int counter = 0;
        int len = ui.listWidget->count();
        for (int i = 0;i < len;i++) {
            if(ui.listWidget->item(i)->checkState() == Checked)
                counter++;
        }

        if (counter > MEDIUM)
            item->setCheckState(Unchecked);
    }

    else if(ui.comboBox_complexity->currentText().compare("High") == 0) {
        item->setCheckState(Checked);


    }
    else if (ui.comboBox_complexity->currentText().compare("Low") == 0) {
        int counter = 0;
        int len = ui.listWidget->count();
        for (int i = 0;i < len;i++) {
            if(ui.listWidget->item(i)->checkState() == Checked)
                counter++;
        }

        if (counter > LOW)
            item->setCheckState(Unchecked);
    }

}

void randomGenerator_GUI::MainWindow::on_comboBox_complexity_currentIndexChanged(const QString &arg1)
{
    if (arg1.compare("High") == 0) {
        int len = ui.listWidget->count();
        for (int i = 0;i < len;i++)
            ui.listWidget->item(i)->setCheckState(Checked);
    }
    else if (arg1.compare("Medium") == 0) {
        int len = ui.listWidget->count();
        for (int i = 0;i < len;i++)
            ui.listWidget->item(i)->setCheckState(Unchecked);
    }
    else if (arg1.compare("Low") == 0) {
        int len = ui.listWidget->count();
        for (int i = 0;i < len;i++)
            ui.listWidget->item(i)->setCheckState(Unchecked);
    }

}

void randomGenerator_GUI::MainWindow::on_pushButton_UncheckAll_clicked()
{
    int len = ui.listWidget->count();
    for (int i = 0;i < len;i++)
        ui.listWidget->item(i)->setCheckState(Unchecked);
}

void randomGenerator_GUI::MainWindow::on_pushButton_Generate_clicked()
{
    if(ui.spinBox->value() <= 0) {
        ui.label_status->setText("<font color='red'>Status: Increase number of manipulating items</font>");
        return;
    }

    else if(ui.comboBox_Test->currentText().compare("BMT") == 0) {
        if (ui.comboBox_complexity->currentText().compare("High") == 0) {
            int len = ui.listWidget->count();

            for (int i = 0;i < len;i++) {
                if(ui.listWidget->item(i)->checkState() == Unchecked) {
                    ui.label_status->setText("<font color='red'>Status: Not all items are checked</font>");
                    return;
                }
            }
            if(ui.spinBox->value() > 12) {
                ui.label_status->setText("<font color='red'>Status: Reduce number of manipulating items</font>");
                return;
            }

        }
        else  if (ui.comboBox_complexity->currentText().compare("Medium") == 0) {
            int len = ui.listWidget->count();
            int counter = 0;

            for (int i = 0;i < len;i++) {
                if(ui.listWidget->item(i)->checkState() == Checked) {
                    counter++;
                }
            }
            if (counter < MEDIUM ) {
                ui.label_status->setText(QString("<font color='red'>Status: Less than %1 items are checked</font>").arg(QString::number(MEDIUM)));
                return;
            }
            else if (counter > MEDIUM) {
                ui.label_status->setText(QString("<font color='red'>Status: More than %1 items are checked</font>").arg(QString::number(MEDIUM)));
                return;
            }
            else if (ui.spinBox->value() > MEDIUM) {
                ui.label_status->setText("<font color='red'>Status: Reduce number of manipulating items</font>");
                return;
            }
        }
        else if (ui.comboBox_complexity->currentText().compare("Low") == 0) {
            int len = ui.listWidget->count();
            int counter = 0;

            for (int i = 0;i < len;i++) {
                if(ui.listWidget->item(i)->checkState() == Checked) {
                    counter++;
                }
            }
            if (counter < LOW ) {
                ui.label_status->setText(QString("<font color='red'>Status: Less than %1 items are checked</font>").arg(QString::number(LOW)));
                return;
            }
            else if (counter > LOW) {
                ui.label_status->setText(QString("<font color='red'>Status: More than %1 items are checked</font>").arg(QString::number(LOW)));
                return;
            }
            else if (ui.spinBox->value() > LOW) {
                ui.label_status->setText("<font color='red'>Status: Reduce number of manipulating items</font>");
                return;
            }
        }

        ui.label_status->setText("<font color='green'>Status: OK</font>");

        rand->init();
        int len = ui.listWidget->count();

        for (int i = 0;i < len;i++) {
            if(ui.listWidget->item(i)->checkState() == Checked) {
                rand->addItem(ui.listWidget->item(i)->text().toStdString());
            }
        }


        int row = ui.spinBox->value();

        std::string orientation, object, pos;

        for(int i = 0; i< row; i++) {
            rand->getRandObjectList(orientation, object, pos);
            if(ui.tableWidget_Result->item(i,0) != 0)
                ui.tableWidget_Result->item(i,0)->setText(object.c_str());
            if(ui.tableWidget_Result->item(i,1) != 0)
                ui.tableWidget_Result->item(i,1)->setText(orientation.c_str());
            if(ui.tableWidget_Result->item(i,2) != 0)
                ui.tableWidget_Result->item(i,2)->setText(pos.c_str());

        }
    }
    else if(ui.comboBox_Test->currentText().compare("BNT") == 0) {
        if( ui.spinBox->value() > 12) {
            ui.label_status->setText("<font color='red'>Status: Reduce number of manipulating items</font>");
            return;
        }

        ui.label_status->setText("<font color='green'>Status: OK</font>");

        std::string path;
        if(!ros::package::getPath("randomGenerator_GUI").empty())
            path = ros::package::getPath("randomGenerator_GUI");
        else
            path = "..";

        std::stringstream ss_taglist;
        std::string file_taglist;
        ss_taglist << path << "/tagList.txt";
//        ss_taglist << "../tagList.txt";
        ss_taglist >> file_taglist;
        char * buffer_taglist = new char[file_taglist.length()];
        strcpy(buffer_taglist,file_taglist.c_str());
        std::fstream f_taglist;
        f_taglist.open(buffer_taglist, ios::in);
        if(f_taglist) {
            string objectN;
            while(!f_taglist.eof()) {
                f_taglist >> objectN;
                rand->addTag(objectN);
            }
        }

        int row = ui.spinBox->value();

        std::string direction, tag;

        for(int i = 0; i< row; i++) {
            rand->getRandTagList(direction, tag);
            if(ui.tableWidget_Result->item(i,0) != 0)
                ui.tableWidget_Result->item(i,0)->setText(tag.c_str());
            if(ui.tableWidget_Result->item(i,1) != 0)
                ui.tableWidget_Result->item(i,1)->setText(direction.c_str());
            if(ui.tableWidget_Result->item(i,2) != 0)
                ui.tableWidget_Result->item(i,2)->setText("--");

        }
    }

}

void randomGenerator_GUI::MainWindow::on_pushButton_Delete_clicked()
{
    for(int i = 0; i< ui.spinBox->value(); i++) {
        ui.tableWidget_Result->item(i,0)->setText("");
        ui.tableWidget_Result->item(i,1)->setText("");
        ui.tableWidget_Result->item(i,2)->setText("");
    }


}

void randomGenerator_GUI::MainWindow::on_pushButton_Save_clicked()
{

    if (ui.lineEdit_teamName->text().compare("Please insert team name") != 0 && !ui.lineEdit_teamName->text().isEmpty()) {

        ui.label_status->setText("<font color='green'>Status: OK</font>");

        std::string path;
        if(!ros::package::getPath("randomGenerator_GUI").empty())
            path = ros::package::getPath("randomGenerator_GUI");
        else
            path = "..";

        std::stringstream ss;
        ss << path << "/taskLog_" << ui.lineEdit_teamName->text().toStdString()  << ".txt";
//        ss << "../taskLog_" << ui.lineEdit_teamName->text().toStdString()  << ".txt";
        std::string file;
        ss >> file;
        char * buffer = new char[file.length()];
        strcpy(buffer,file.c_str());
        std::fstream f;
        f.open(buffer, ios::app | ios::out);
        f << ui.comboBox_Test->currentText().toStdString();
        if(ui.comboBox_Test->currentText().compare("BMT") == 0)
            f << "-" << ui.comboBox_complexity->currentText().toStdString();
        f << endl;
        for(int i = 0; i< ui.tableWidget_Result->rowCount(); i++) {
            std::string temp_object = ui.tableWidget_Result->item(i,0)->text().toStdString();
            std::string temp_orientation = ui.tableWidget_Result->item(i,1)->text().toStdString();
            std::string temp_position = ui.tableWidget_Result->item(i,2)->text().toStdString();
            f << i << ":\t" << temp_object << "\t  " << temp_orientation << "\t  " << temp_position << endl;
        }

        f.close();
    }
    else {
        ui.lineEdit_teamName->setText("Please insert team name");
        ui.label_status->setText("<font color='red'>Status: No team name</font>");
    }

}



void randomGenerator_GUI::MainWindow::on_comboBox_Test_currentIndexChanged(const QString &arg1)
{
    if (arg1.compare("BNT") == 0) {
        ui.tableWidget_Result->horizontalHeaderItem(0)->setText("Tag");
    }
    else if (arg1.compare("BMT") == 0 || arg1.compare("PPT") == 0) {
        ui.tableWidget_Result->horizontalHeaderItem(0)->setText("Object");
    }
}

void randomGenerator_GUI::MainWindow::on_spinBox_valueChanged(int arg1)
{
    ui.tableWidget_Result->setRowCount(arg1);

    for(int i = 0; i < arg1; i++) {
        if(ui.tableWidget_Result->item(i,0) == 0)
            ui.tableWidget_Result->setItem(i,0,new QTableWidgetItem());
        if(ui.tableWidget_Result->item(i,1) == 0)
            ui.tableWidget_Result->setItem(i,1,new QTableWidgetItem());
        if(ui.tableWidget_Result->item(i,2) == 0)
            ui.tableWidget_Result->setItem(i,2,new QTableWidgetItem());
    }
}
