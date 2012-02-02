/**
 * \file correspondances.h
 * \brief Header for matching
 *
 * Empty.
 *
 */

#ifndef MATCH
#define MATCH
#include <iostream>

#include "descriptor.h"

/// A vector of descriptor for matching
typedef std::pair<descriptor*,descriptor*> pairMatch;
typedef std::vector<pairMatch > listMatch;
float euclideanDistance(descriptor *a,descriptor* b);

image*	 showDescriptors(image* img1,listDescriptor* listeDesc,bool afficher);
void lign(image *img,float xa,float ya,float xb, float yb,float intensite);
listMatch* matchDescriptor(listDescriptor * l1, listDescriptor * l2);

#endif