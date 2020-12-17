# MVKM-Java
Java implementation of the Multi-View Knowldge Model

The following describes how to set up a project in Eclipse and build it successfully

## Import to Eclipse

There are two options to import MVKM-Java into Eclipse: using a Git [fork](https://help.github.com/articles/fork-a-repo) or using a downloaded package. If you are not familiar with Git or GitHub, you should opt for the downloaded package.

### Option A: GitHub

1. Fork the repository to use as a starting point.
    * Navigate to https://github.com/wdahl/MVKM-Java in your browser.
    * Click the "Fork" button in the top-right of the page.
    * Once your fork is ready, open the new repository's "Settings" by clicking the link in the menu bar on the right.
    * NOTE: GitHub only allows you to fork a project once. If you need to create multiple forks, you can follow these [instructions](http://adrianshort.org/2011/11/08/create-multiple-forks-of-a-github-repo/).
2. Clone your new repository to your Eclipse workspace.
    * Open Eclipse and select the File → Import... menu item.
    * Select Git → Projects from Git, and click "Next >".
    * Select "URI" and click "Next >". 
    * Enter your repository's clone URL in the "URI" field. The remaining fields in the "Location" and "Connection" groups will get automatically filled in.
    * Enter your GitHub credentials in the "Authentication" group, and click "Next >".
    * Select the `master` branch on the next screen, and click "Next >".
    * The default settings on the "Local Configuration" screen should work fine, click "Next >".
    * Make sure "Import existing projects" is selected, and click "Next >".
    * Eclipse should find and select the `MVKM-Java` automatically, click "Finish".
  
### Option B: Downloaded Package

1. Download the project [here](https://github.com/wdahl/MVKM-Java). **Don't unzip the ZIP file yet.**
2. Create a new Java project in Eclipse. 
    * From the menubar choose File → New → Java Project. 
    * Give the project the name of your tool.
    * Click "Finish".
3. Import the source files.
    * Right-click (ctrl-click) onto the folder icon of your newly created project in the Package Explorer and select "Import..."`" from the menu that pops up. 
    * Select General → Archive File, and click "Next >".
    * Navigate to the ZIP file you downloaded earlier in step 1, and click "Finish".
