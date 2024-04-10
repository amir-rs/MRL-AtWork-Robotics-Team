/**
 * @file /include/randomGenerator_GUI/qnode.hpp
 *
 * @brief Communications central!
 *
 * @date February 2011
 **/
/*****************************************************************************
** Ifdefs
*****************************************************************************/

#ifndef randomGenerator_GUI_QNODE_HPP_
#define randomGenerator_GUI_QNODE_HPP_

/*****************************************************************************
** Includes
*****************************************************************************/

#include <ros/ros.h>
#include <string>
#include <QThread>
#include <QStringListModel>


/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace randomGenerator_GUI {

/*****************************************************************************
** Class
*****************************************************************************/

class QNode : public QThread {
    Q_OBJECT
public:
	QNode(int argc, char** argv );
	virtual ~QNode();
	bool init();
	bool init(const std::string &master_url, const std::string &host_url);
	void run();

Q_SIGNALS:
    void rosShutdown();

private:
	int init_argc;
	char** init_argv;
};

}  // namespace randomGenerator_GUI

#endif /* randomGenerator_GUI_QNODE_HPP_ */
