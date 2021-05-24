/* fof - a C file for doing friends-of-friends and probability friends-of-friends group identification.
 * Zackary L. Hutchens, UNC Chapel Hill, July 2019.
 *
 */

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define LINE_SIZE 10000
#define H0 100.0
#define c 3.00e+05
#define PI 3.14159265359
#define MAX_GALAXIES 20000
#define STEPS 1000

typedef struct GalaxyObject {
	char name[10];
	double ra;
	double dec;
	double mag;
	double cz;
	int groupID;
	double czerr;
} Galaxy;

typedef struct GroupObject {
	int groupID;
	Galaxy members[300];
} Group;


// function prototypes
void reset_groups(Galaxy gxs[], int lengxs);
void prob_fof(Galaxy gxs[], int lengxs,  double bperp, double blos, double s, float Pth);
int prob_linking_length(Galaxy g1, Galaxy g2, double b_perp, double b_los, double s, float Pth);
double ang_sep(Galaxy g1, Galaxy g2);
double d_perp(Galaxy g1, Galaxy g2);
double gauss(double x, double mu, double sigma);
double inside_int(double z, double VL, Galaxy g2);
double pfof_integrand(double z, double VL, Galaxy g1, Galaxy g2);
double int_simpson(double from, double to, double n, double VL, Galaxy g1, Galaxy g2, double (*func)(double z, double VL, Galaxy g1, Galaxy g2));


// start program
int main(void)
{
   fprintf(stdout, "Warning: this code is still under development and the numerical integration may not be accurate.")
   // Read in the filename and assign name to a pointer.
   FILE *fptr;
   char filename[75];
   fprintf(stdout, "Enter a filename to read: ");
   fscanf(stdin, "%s", filename);

   if ((fptr = fopen(filename, "r"))==NULL){
	   fprintf(stderr, "Cannot open specified file.\n");
	   exit(1);
   }
   rewind(fptr);

   // Read file line-by-line, and initialize galaxies into an array.
   char line[LINE_SIZE];
   int i=0;
   double ra, dec, mag, cz, czerr;
   Galaxy temp;
   char name[10];

   Galaxy galaxy_arr[MAX_GALAXIES];
   while (fgets(line, sizeof(line), fptr) != NULL) {
      if (i != 0) {
         sscanf(line, "%s\t%lf\t%lf\t%lf\t%lf\t%lf\n", name, &ra, &dec, &mag, &cz, &czerr);
         strcpy(temp.name, name);
         temp.ra = ra;
         temp.dec = dec;
         temp.mag = mag;
         temp.cz = cz;
         temp.czerr = czerr;
	 temp.groupID = 0; // default all group ID's to zero
         galaxy_arr[i-1] = temp; 
         i++;
      } else {
	 i++;
      }
   }
   int number_of_galaxies = i-1;
   fclose(fptr);

   // Prompt user for the volume, bperp, blos
   double volume, bperp, blos, Pth;
   int N;
   fprintf(stdout, "The data has been read. Enter the volume of the sample in Mpc/h: ");
   fscanf(stdin, "%lf", &volume);
   
   fprintf(stdout, "Enter the number of galaxies by which the separation will be computed: ");
   fscanf(stdin, "%d", &N);

   fprintf(stdout, "Enter the line-of-sight linking constant (use 1.1 for ECO/RESOLVE-B): ");
   fscanf(stdin, "%lf", &blos);
   
   fprintf(stdout, "Enter the perpendicular linking constant (use 0.7 for ECO/RESOLVE-B): ");
   fscanf(stdin, "%lf", &bperp);

   fprintf(stdout, "Enter the threshold probability: ");
   fscanf(stdin, "%lf", &Pth);

   if ((Pth < 0) || (Pth >= 1)){
      fprintf(stderr, "Threshold probability must be between 0 and 1\n");
      exit(1);
   }
    
   clock_t begin = clock();
   // Now that you have an array of galaxies + all quantities, do the group finding.
   double s = pow((N/volume), -1.0/3.0); 
   fprintf(stdout, "Mean separation between galaxies: %lf Mpc/h\n", s);
   
   prob_fof(galaxy_arr, number_of_galaxies, bperp, blos, s, Pth); 

   // Output galaxies to file with group ID numbers
   FILE *sptr;
   char savename[75];
   fprintf(stdout, "Enter a file destination for the new galaxy group catalog: ");
   fscanf(stdin, "%s", savename);

   sptr = fopen(savename, "w");

   char outline[100] = "Name\tRA\tDec\tMagnitude\tcz\tczerr\tGroupID\n";
   fwrite(outline, 1, sizeof(outline), sptr);
   for (int k=0; k < number_of_galaxies; k++){
      sprintf(outline, "%s\t%lf\t%lf\t%lf\t%lf\t%lf\t%d\n", galaxy_arr[k].name, galaxy_arr[k].ra, galaxy_arr[k].dec, galaxy_arr[k].mag, galaxy_arr[k].cz, galaxy_arr[k].czerr, galaxy_arr[k].groupID);
      fwrite(outline, 1, sizeof(outline), sptr);
   }
   clock_t end = clock();
   double time_spent = (double)(end-begin) / CLOCKS_PER_SEC;
   fprintf(stdout, "Execution Time: %lf", time_spent);
   return 0;    
}

// functions +


void prob_fof(Galaxy gxs[], int lengxs, double bperp, double blos, double s, float Pth){
   /* prob_fof - perform a probability friends-of-friends analysis on a group of galaxies.
    * Arguments:
    * 	gxs - array of galaxies (type Galaxy)
    * 	lengxs - number of galaxies in gxs
    * 	bperp - perpendicular linking constant
    * 	blos - line-of-sight linking constant
    * 	s - mean separation between galaxies.
    * 	Pth - threshold probability for what constitutes galaxy friendship (must be on range 0,1).
    * Returns:
    *   None. Sets every galaxy to have a unique, non-zero group ID number.
    */
   int grpindex = 1;
   reset_groups(gxs, lengxs);
   
   int keeping_id, equiv_id;

   for (int i=0; i <= lengxs; i++){
      for (int j=0; j <= lengxs; j++){
         if (strcmp(gxs[i].name, gxs[j].name) != 0){
            if (prob_linking_length(gxs[i], gxs[j], bperp, blos, s, Pth)){
               
	       if ((gxs[i].groupID == gxs[j].groupID) && (gxs[i].groupID != 0) && (gxs[j].groupID)){   // case 1: already in same group
		   continue;
	       } else if ((gxs[i].groupID != gxs[j].groupID) && (gxs[i].groupID != 0) && (gxs[j].groupID == 0)) { // case 2: g1 in group, g2 is not.
                   gxs[j].groupID = gxs[i].groupID;
	       } else if ((gxs[i].groupID != gxs[j].groupID) && (gxs[i].groupID == 0) && (gxs[j].groupID != 0)) { // case 3: g2 in group, g1 is not.
                   gxs[i].groupID = gxs[j].groupID;
	       } else if ((gxs[i].groupID == 0) && (gxs[j].groupID == 0)) {  // case 4: neither in a group
                   gxs[j].groupID = grpindex;
		   gxs[i].groupID = grpindex;
		   grpindex++;
	       } else if ((gxs[i].groupID != gxs[j].groupID) && (gxs[i].groupID != 0) && (gxs[j].groupID != 0)) { // case 5: both already in groups, but their groups need to be merged.
                   keeping_id = gxs[i].groupID;
		   equiv_id = gxs[j].groupID;

		   for (int k=0; k <= lengxs; k++){
                      if (gxs[i].groupID == equiv_id){
                         gxs[i].groupID = keeping_id;
		      }
		   }
	           
	      } else {
                  // Should never get here
		  fprintf(stderr, "Galaxies %s and %s failed the PFoF algorithm\n", gxs[i].name, gxs[j].name);
		  exit(1);
	      }
           }
        }
     }
   }

   // assign group ID's to all the single-galaxy groups
   for (int m=0; m <= lengxs; m++){
      if (gxs[m].groupID == 0){
         gxs[m].groupID = grpindex + m;
      }
   }
   printf("PFoF group finding complete.\n");
}


void reset_groups(Galaxy gxs[], int lengxs){
   /* reset_groups - reset every galaxy in an array to have groupID = 0.
    * Arguments:
    * 	gxs - array of galaxies (type Galaxy)
    * 	lengxs - number of galaxies in gxs
    * Returns:
    *	None - reset all groupID's to zero.
    *
    */
   for (int i=0; i < lengxs; i++){
      gxs[i].groupID = 0;
   }
}

int prob_linking_length(Galaxy g1, Galaxy g2, double b_perp, double b_los, double s, float Pth){
   /* prob_linking_length: determine if two galaxies are friends in a probability friends-of-friends scheme.
    * Arguments:
    * 	g1, g2 - galaxies to compare, type Galaxy
    * 	b_perp - perpendicular linking constant
    * 	b_los - line-of-sight linking constant
    * 	s - mean separation bewteen galaxies
    * 	Pth - threshold probability for what constitutes group friendship.
    * Returns:
    *   Bool indicating whether the galaxies are friends.
    */
    
   // test perpendicular linking condition
   double DPERP = d_perp(g1, g2); 
   
   // test line-of-sight condition. Integrate up to + around + after the distribution in adaptive steps
   double val1 = int_simpson(0.0, g1.cz/c - 3*g1.czerr/c, 10000, b_los*s, g1, g2, pfof_integrand); 
   double val2 = int_simpson(g1.cz/c-3*g1.czerr/c, g1.cz/c+3*g1.czerr/c, 510000, b_los*s, g1, g2, pfof_integrand);
   double val3 = int_simpson(g1.cz/c + 3*g1.czerr/c, 20.0, 10000, b_los*s, g1, g2, pfof_integrand);

   double P = val1 + val2 + val3;
   if ((DPERP <= b_perp * s) && (P >= Pth)){  
      return 1;
   } else {
      return 0;
   }
}


double d_perp(Galaxy g1, Galaxy g2){
   /* d_perp - compute the perpendicular distance between galaxies.
    * Arguments:
    * 	g1, g2 - galaxies (type Galaxy) to compute sky distance
    * Returns:
    *   DPERP (type double), on-sky distance between them in Mpc/h.
    */
    double thetaij = ang_sep(g1, g2);
    return (1/H0)*(g1.cz + g2.cz) * sin(thetaij / 2.0);
}

double ang_sep(Galaxy g1, Galaxy g2){
   /* ang_sep - compute the angular separation between two galaxies using the Haversine formula.
    * Arguments:
    * 	g1, g2 - galaxies (type Galaxy) to compute angle
    * Returns:
    * 	havtheta (type double), separation between galaxies in radians.
    */
    double phi1 = g1.ra * PI/180.0;
    double phi2 = g2.ra * PI/180.0;
    double theta1 = PI/2.0 - g1.ra*PI/180.0;
    double theta2 = PI/2.0 - g2.ra*PI/180.0;

    double havtheta = 2*asin(sqrt(pow(sin((theta2-theta1)/2.0),2.0) + sin(theta2)*sin(theta1)*pow(sin((phi2-phi1)/2.0),2.0)));
    return havtheta;
}

double gauss(double x, double mu, double sigma){
  /* Gaussian function
   * Arguments: x, mu, sigma
   * Returns: PDF evaluated at `x`
   */
   return (1.0/(sqrt(2*PI)*sigma) * exp(-1.0 * 0.5 * pow(((x-mu)/sigma), 2.0)));
}

double inside_int(double z, double VL, Galaxy g2){
   /* inside_int - inside integral for PFoF (analytically computed)
    * Arguments: redshift z, linking length VL, galaxy g2
    * Returns: value of inside integral at z; F(z).
    */
    return (0.5*erf((z+VL-g2.cz/c)/(sqrt(2)*g2.czerr/c)) - 0.5*erf((z-VL-g2.cz/c)/(sqrt(2)*g2.czerr/c)));
}

double pfof_integrand(double z, double VL, Galaxy g1, Galaxy g2){
   /* pfof_integrand - integrand needed to be integrated for PFoF
    * Arguments: redshift z, linking length VL, Galaxies g1 + g2
    * Returns: Integrand evaluated at `z`
    */
   return gauss(z, g1.cz/c, g1.czerr/c) * inside_int(z, VL, g2);
}

double int_simpson(double from, double to, double n, double VL, Galaxy g1, Galaxy g2, double (*func)(double z, double VL, Galaxy g1, Galaxy g2))
{
   double h = (to - from) / n;
   double sum1 = 0.0;
   double sum2 = 0.0;
   int i;

   for(i = 0;i < n;i++)
      sum1 += (*func)(from + h * i + h / 2.0, VL, g1, g2);

   for(i = 1;i < n;i++)
      sum2 += (*func)(from + h * i, VL, g1, g2);

   return h / 6.0 * ((*func)(from, VL, g1, g2) + (*func)(to, VL, g1, g2) + 4.0 * sum1 + 2.0 * sum2); // this could be wrong?
}
